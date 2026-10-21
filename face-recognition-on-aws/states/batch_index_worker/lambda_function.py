"""
StateBatchIndexWorker — Map worker for batch face indexing (README.md §19).

Per iterator: process manifest lines in grid batches —
  1. Download + magic-byte sniff (failed/unsupported_format before decode).
  2. prepare_tile (fit-inside 480x640, letterboxed; placement recorded).
     Decode failure -> failed/decode_error (HEVC-less HEIC etc).
  3. compose_grid (~80 tiles, separators) -> ONE IndexFaces call
     (QualityFilter AUTO, batch-level ExternalImageId).
  4. largest_face_per_cell: per tile the largest mapped face wins; every
     other returned FaceId (phantoms, bleed) is deleted from the collection.
  5. Lines whose tile produced no face: ONE solo IndexFaces retry at gated
     full resolution (fit-inside BATCH_SOLO_MAX_EDGE_PX) — full-body shots
     whose face went sub-40px in the tile — before failed/no_face.
  6. Per indexed face: store derivatives (thumbnail crop + downscaled WHOLE
     source image) + register DDB record (origin=user, source_key/etag,
     normalized coord).

Ordering rule (crash safety, same as search_index_faces): the DDB record is
registered IMMEDIATELY after its IndexFaces succeeds and BEFORE any other
work; the iterator checkpoint is written after every grid. A crash between
IndexFaces and register would orphan collection faces — the checkpoint window
is kept minimal and re-entry skips completed lines via results.

Semaphore: held by the SM around the Map (IndexFaces shares the Rekognition
TPS pool with recognize_faces runs); the worker refreshes its holder TTL.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from lib import semaphore
from lib.custom_exceptions import LambdaResourceTimeoutException
from lib.face_store import register_face, store_face_images
from lib.grid_image import compose_grid, downscale_fit
from lib.index_batch import (
    create_batch_external_id,
    format_accepted,
    largest_face_per_cell,
    prepare_tile,
    sniff_format,
    tile_box_to_source,
)
from lib.input_schema import (
    BATCH_INDEX_QUALITY_FILTER,
    BATCH_SOLO_MAX_EDGE_PX,
    BATCH_TILE_COLS,
    BATCH_TILE_DIM,
    BATCH_TILES_PER_GRID,
    REMAINING_TIME_BUFFER_MS,
)
from lib.rekognition_utils import CallCounter, delete_faces, index_faces
from lib.s3_utils import (
    delete_keys,
    get_bytes,
    get_json,
    list_keys,
    parse_s3uri,
    put_bytes,
    put_json,
)

logger = logging.getLogger(__name__)

_DOWNLOAD_THREADS = 8


def lambda_handler(event: dict, context) -> dict:
    item = event["item_data"]
    ctx = item["context"]
    out_bucket = ctx["out_bucket"]
    collection_id = ctx["collection_id"]
    execution_arn = event.get("execution_arn") or ""
    if execution_arn:
        semaphore.refresh_holder(execution_arn)

    iterator = get_json(out_bucket, item["key"])
    results: dict = iterator.get("results") or {}
    counter = CallCounter()
    counter.add(iterator.get("usage") or {})

    out_prefix = parse_s3uri(ctx["output_s3uri"])[1].rstrip("/")
    grids_prefix = f"{out_prefix}/grids"
    facestore_prefix = (f"micro-workflow/face-recognition/_facestore/"
                        f"{collection_id}")
    batch_id = event.get("uuid") or "batch"

    pending = [e for e in iterator["entries"]
               if e["image_uri"] not in results]

    # ── Re-entry crash repair (double-index guard) ────────────────────────────
    # register_face runs per LINE but the S3 checkpoint lands per GRID: a
    # hard crash mid-grid (OOM, hard timeout past the buffer) leaves lines
    # indexed+registered in DDB that the stale checkpoint doesn't know about.
    # Re-running them through IndexFaces would mint DUPLICATE FaceIds. DDB is
    # the ground truth: on re-entry with pending work, look up this
    # iterator's source_etags and absorb already-registered lines into
    # results instead of re-indexing.
    if pending and results:
        registered = _registered_etags(
            collection_id, {e["source_etag"] for e in pending})
        still_pending = []
        for e in pending:
            hit = registered.get(e["source_etag"])
            if hit:
                results[e["image_uri"]] = {
                    "image_uri": e["image_uri"],
                    "celebrity_name": e["celebrity_name"],
                    "status": "indexed", "face_id": hit,
                    "reason": "recovered_on_reentry"}
            else:
                still_pending.append(e)
        if len(still_pending) != len(pending):
            logger.info("re-entry: recovered %d already-registered lines",
                        len(pending) - len(still_pending))
        pending = still_pending

    # O(n) side-object checkpoints (same shape as the detect workers): each
    # grid's results land in {iterator_key}.grid_NNNN.json once; the full
    # iterator is rewritten only at consolidation. The old per-grid full
    # rewrite made total checkpoint bytes O(n^2) with manifest size.
    for key in sorted(list_keys(out_bucket, item["key"] + ".grid_")):
        side = get_json(out_bucket, key)
        for uri, res in (side.get("results") or {}).items():
            if uri not in results:
                results[uri] = res
        counter.add(side.get("usage") or {})

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_THREADS) as pool:
        for gi in range(0, len(pending), BATCH_TILES_PER_GRID):
            batch = [e for e in pending[gi:gi + BATCH_TILES_PER_GRID]
                     if e["image_uri"] not in results]
            if not batch:
                continue
            if context.get_remaining_time_in_millis() < REMAINING_TIME_BUFFER_MS:
                # Side objects already persist completed grids.
                raise LambdaResourceTimeoutException(
                    f"batch index checkpointed at entry {gi}")

            grid_results: dict = {}
            grid_counter = CallCounter()
            _process_grid(batch, pool, out_bucket, grids_prefix,
                          facestore_prefix, collection_id, batch_id,
                          item["item_id"], gi // BATCH_TILES_PER_GRID,
                          grid_counter, grid_results)
            put_json(out_bucket,
                     f"{item['key']}.grid_{gi // BATCH_TILES_PER_GRID:04d}.json",
                     {"results": grid_results,
                      "usage": grid_counter.as_dict()})
            results.update(grid_results)
            counter.add(grid_counter.as_dict())

    # Consolidate: one full write, then drop the side objects.
    iterator["results"] = results
    iterator["usage"] = counter.as_dict()
    put_json(out_bucket, item["key"], iterator)
    side = list_keys(out_bucket, item["key"] + ".grid_")
    if side:
        delete_keys(out_bucket, sorted(side))

    return {"item_id": item["item_id"], "key": item["key"],
            "usage": counter.as_dict()}


def _registered_etags(collection_id: str, etags: set) -> dict:
    """source_etag -> face_id for etags already registered in DDB (crash
    repair on re-entry). Queries the collection-index GSI once."""
    import boto3
    from lib.face_store import TABLE_NAME, unmarshal
    from lib.input_schema import COLLECTION_INDEX_GSI
    ddb = boto3.client("dynamodb")
    found: dict = {}
    kwargs: dict = {
        "TableName": TABLE_NAME,
        "IndexName": COLLECTION_INDEX_GSI,
        "KeyConditionExpression": "collection_id = :c",
        "ProjectionExpression": "face_id, source_etag",
        "ExpressionAttributeValues": {":c": {"S": collection_id}},
    }
    while True:
        resp = ddb.query(**kwargs)
        for raw in resp.get("Items") or []:
            rec = unmarshal(raw)
            if rec.get("source_etag") in etags:
                found[rec["source_etag"]] = rec["face_id"]
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            return found
        kwargs["ExclusiveStartKey"] = lek


def _process_grid(batch, pool, out_bucket, grids_prefix, facestore_prefix,
                  collection_id, batch_id, item_id, grid_no,
                  counter, results) -> None:
    # ── Download + sniff + tile (threaded) ────────────────────────────────────
    def _load(entry):
        uri = entry["image_uri"]
        try:
            b, k = parse_s3uri(uri)
            data = get_bytes(b, k)
        except Exception as exc:                              # noqa: BLE001
            return entry, None, None, f"download_error:{type(exc).__name__}"
        fmt = sniff_format(data)
        if not format_accepted(fmt):
            return entry, None, None, f"unsupported_format:{fmt or 'unknown'}"
        try:
            tile, placement = prepare_tile(data)
        except Exception as exc:                              # noqa: BLE001
            return entry, None, None, f"decode_error:{type(exc).__name__}"
        return entry, data, (tile, placement), None

    loaded = list(pool.map(_load, batch))

    tiles, tile_entries = [], []
    for entry, data, prepared, err in loaded:
        line = {"image_uri": entry["image_uri"],
                "celebrity_name": entry["celebrity_name"]}
        if err:
            results[entry["image_uri"]] = {**line, "status": "failed",
                                           "reason": err}
            continue
        tiles.append(prepared[0])
        tile_entries.append((entry, data, prepared[1]))

    if not tiles:
        return

    # ── One IndexFaces per grid ───────────────────────────────────────────────
    # 1080px tiles measured ~6MB/81-tile grid on real photos; extremely
    # grainy/noisy content could still trip the 14 MB JPEG bound. Degrade
    # toward MORE calls, never toward failing lines: split the batch in half
    # and grid each half (recursion bottoms out at a single tile, whose JPEG
    # is by construction far below the bound).
    tile_w, tile_h = BATCH_TILE_DIM
    cols = BATCH_TILE_COLS
    rows = -(-len(tiles) // cols)  # ceil
    try:
        grid_jpeg = compose_grid(tiles, cols, rows, tile_w, tile_h)
    except RuntimeError:
        if len(tile_entries) == 1:
            raise  # single tile over 14 MB — genuinely broken input
        half = len(tile_entries) // 2
        sub = [e for e, _, _ in tile_entries]
        logger.info("grid JPEG over bound with %d tiles — splitting",
                    len(tile_entries))
        # Suffix child labels ("3a"/"3b", recursively "3aa"...) — an
        # arithmetic scheme (grid_no*100+1) collided with REAL grid numbers,
        # silently overwriting other grids' audit JPEGs.
        _process_grid([dict(e) for e in sub[:half]], pool, out_bucket,
                      grids_prefix, facestore_prefix, collection_id,
                      batch_id, item_id, f"{grid_no}a", counter, results)
        _process_grid([dict(e) for e in sub[half:]], pool, out_bucket,
                      grids_prefix, facestore_prefix, collection_id,
                      batch_id, item_id, f"{grid_no}b", counter, results)
        return
    grid_key = f"{grids_prefix}/batch_{item_id:03d}_{grid_no}.jpg"
    put_bytes(out_bucket, grid_key, grid_jpeg, "image/jpeg")

    external_id = create_batch_external_id(batch_id)
    records = index_faces(collection_id, out_bucket, grid_key, external_id,
                          max_faces=100,
                          quality_filter=BATCH_INDEX_QUALITY_FILTER,
                          counter=counter)

    winners = largest_face_per_cell(records, len(tiles), cols, rows)
    winner_ids = {w["record"]["Face"]["FaceId"] for w in winners.values()}
    phantoms = [r["Face"]["FaceId"] for r in records
                if r["Face"]["FaceId"] not in winner_ids]
    if phantoms:
        delete_faces(collection_id, phantoms)
        logger.info("grid %s: %d phantoms deleted", grid_key, len(phantoms))

    # ── Per-line outcome ──────────────────────────────────────────────────────
    for idx, (entry, data, placement) in enumerate(tile_entries):
        line = {"image_uri": entry["image_uri"],
                "celebrity_name": entry["celebrity_name"]}
        win = winners.get(idx)
        if win is None:
            # Tiny-face fallback: solo full-resolution IndexFaces.
            outcome = _solo_index(entry, data, collection_id, external_id,
                                  out_bucket, facestore_prefix, counter)
            results[entry["image_uri"]] = {**line, **outcome}
            continue

        face_id = win["record"]["Face"]["FaceId"]
        coord = tile_box_to_source(win["box"], placement)
        _store_and_register(face_id, entry, data, coord, collection_id,
                            external_id, out_bucket, facestore_prefix)
        results[entry["image_uri"]] = {**line, "status": "indexed",
                                       "face_id": face_id}


def _solo_index(entry, data, collection_id, external_id,
                out_bucket, facestore_prefix, counter) -> dict:
    """Gated full-resolution retry for a tile that yielded no face."""
    solo = downscale_fit(data, BATCH_SOLO_MAX_EDGE_PX)
    solo_key = (f"{facestore_prefix}/_staging/solo_"
                f"{entry['source_etag']}.jpg")
    put_bytes(out_bucket, solo_key, solo, "image/jpeg")
    records = index_faces(collection_id, out_bucket, solo_key, external_id,
                          max_faces=100,
                          quality_filter=BATCH_INDEX_QUALITY_FILTER,
                          counter=counter)
    if not records:
        return {"status": "failed", "reason": "no_face"}
    # Largest face wins; the rest are phantoms.
    records.sort(key=lambda r: -((r["Face"]["BoundingBox"]["Width"])
                                 * (r["Face"]["BoundingBox"]["Height"])))
    phantoms = [r["Face"]["FaceId"] for r in records[1:]]
    if phantoms:
        delete_faces(collection_id, phantoms)
    face_id = records[0]["Face"]["FaceId"]
    bbox = records[0]["Face"]["BoundingBox"]
    coord = {"l": round(bbox["Left"], 6), "t": round(bbox["Top"], 6),
             "w": round(bbox["Width"], 6), "h": round(bbox["Height"], 6)}
    _store_and_register(face_id, entry, data, coord, collection_id,
                        external_id, out_bucket, facestore_prefix)
    return {"status": "indexed", "face_id": face_id, "reason": "solo_retry"}


def _store_and_register(face_id, entry, data, coord, collection_id,
                        external_id, out_bucket, facestore_prefix) -> None:
    thumb_key, full_key = store_face_images(
        out_bucket, facestore_prefix, face_id, data, coord)
    register_face(face_id, {
        "collection_id": collection_id,
        "external_image_id": external_id,
        "celeb": entry["celebrity_name"],
        "origin": "user",
        "source_key": entry["image_uri"],
        "source_etag": entry["source_etag"],
        "coord": [coord["l"], coord["t"], coord["w"], coord["h"]],
        "thumbnail_key": thumb_key,
        "full_image_key": full_key,
    })
