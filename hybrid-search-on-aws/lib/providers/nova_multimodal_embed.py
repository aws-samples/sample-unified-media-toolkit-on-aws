"""
nova_multimodal_embed.py — text/image embedding via Nova Multimodal Embeddings on Bedrock.

Text request:
  {
    "taskType": "SINGLE_EMBEDDING",
    "singleEmbeddingParams": {
      "embeddingPurpose": "GENERIC_RETRIEVAL",
      "embeddingDimension": dims,
      "text": {"truncationMode": "NONE", "value": text}
    }
  }

Image request replaces "text" with "image": {"detailLevel": "STANDARD_IMAGE", "format": "jpeg", "source": {...}}.
Response: {"embeddings": [{"embeddingType": "TEXT"|"IMAGE", "embedding": [...]}]}

Supported output dims: 256, 384, 1024, 3072.
"""
from __future__ import annotations


def build_request(text: str, dims: int | None = None) -> dict:
    params: dict = {
        "embeddingPurpose": "GENERIC_RETRIEVAL",
        "text": {"truncationMode": "NONE", "value": text},
    }
    if dims is not None:
        params["embeddingDimension"] = dims
    return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}


def build_image_request(image_b64: str, dims: int | None = None) -> dict:
    params: dict = {
        "embeddingPurpose": "GENERIC_RETRIEVAL",
        "image": {
            "detailLevel": "STANDARD_IMAGE",
            "format": "jpeg",
            "source": {"bytes": image_b64},
        },
    }
    if dims is not None:
        params["embeddingDimension"] = dims
    return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}


def build_document_request(text: str | None, image_b64: str | None,
                           dims: int | None = None) -> dict:
    """Nova takes a SINGLE content item per request — no joint text+image
    embedding. When both are supplied the image wins (visual index)."""
    if image_b64:
        return build_image_request(image_b64, dims)
    return build_request(text or "", dims)


def parse_response(result: dict) -> list[float]:
    # {"embeddings": [{"embeddingType": "TEXT", "embedding": [...]}]}
    return result["embeddings"][0]["embedding"]


MODEL_PREFIX = "nova-2-multimodal-embeddings"
