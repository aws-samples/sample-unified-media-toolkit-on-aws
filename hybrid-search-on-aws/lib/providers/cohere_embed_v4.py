"""
cohere_embed_v4.py — text and image embedding via Cohere Embed v4 on Bedrock.

Text request:  {"texts": [text], "input_type": "search_query", "truncate": "END",
                "output_dimension": dims}
Image request: {"images": ["data:image/jpeg;base64,..."], "input_type": "search_query",
                "embedding_types": ["float"], "output_dimension": dims}

The images array requires the full data URI prefix (data:image/jpeg;base64,<b64>).

Response: {"embeddings": {"float": [[...]]} }  or  {"embeddings": [[...]]}

Supported output dims: 256, 512, 1024, 1536.
"""
from __future__ import annotations


def build_request(text: str, dims: int | None = None) -> dict:
    body: dict = {"texts": [text], "input_type": "search_query", "truncate": "END"}
    if dims is not None:
        body["output_dimension"] = dims
    return body


def build_image_request(image_b64: str, dims: int | None = None) -> dict:
    body: dict = {
        "images": [f"data:image/jpeg;base64,{image_b64}"],
        "input_type": "search_query",
        "embedding_types": ["float"],
    }
    if dims is not None:
        body["output_dimension"] = dims
    return body


def build_document_request(text: str | None, image_b64: str | None,
                           dims: int | None = None) -> dict:
    """
    DOCUMENT-side embedding (indexing): input_type search_document, and
    Cohere's mixture mode when BOTH text and an image are supplied (jointly
    embedded — same space as text/image-only vectors). Ports the old M2C
    Node cohereHelper behavior.
    """
    body: dict = {"input_type": "search_document", "truncate": "END"}
    if text and image_b64:
        body["inputs"] = [{"content": [
            {"type": "text", "text": text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}]
        body["embedding_types"] = ["float"]
    elif image_b64:
        body["images"] = [f"data:image/jpeg;base64,{image_b64}"]
        body["embedding_types"] = ["float"]
    else:
        body["texts"] = [text or ""]
    if dims is not None:
        body["output_dimension"] = dims
    return body


def parse_response(result: dict) -> list[float]:
    embeddings = result["embeddings"]
    if isinstance(embeddings, dict):
        embeddings = embeddings.get("float") or next(iter(embeddings.values()))
    return embeddings[0]


MODEL_PREFIX = "cohere.embed"
