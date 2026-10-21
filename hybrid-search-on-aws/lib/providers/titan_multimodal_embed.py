"""
titan_multimodal_embed.py — text/image embedding via Titan Multimodal Embeddings G1 on Bedrock.

Text request:  {"inputText": text, "embeddingConfig": {"outputEmbeddingLength": dims}}
Image request: {"inputImage": <base64>, "embeddingConfig": {"outputEmbeddingLength": dims}}
Response:      {"embedding": [...]}

Supported output dims: 256, 384, 1024.
"""
from __future__ import annotations


def build_request(text: str, dims: int | None = None) -> dict:
    body: dict = {"inputText": text}
    if dims is not None:
        body["embeddingConfig"] = {"outputEmbeddingLength": dims}
    return body


def build_image_request(image_b64: str, dims: int | None = None) -> dict:
    body: dict = {"inputImage": image_b64}
    if dims is not None:
        body["embeddingConfig"] = {"outputEmbeddingLength": dims}
    return body


def build_document_request(text: str | None, image_b64: str | None,
                           dims: int | None = None) -> dict:
    """Titan G1 accepts inputText + inputImage together — native mixture.
    No input_type concept (symmetric model)."""
    body: dict = {}
    if text:
        body["inputText"] = text
    if image_b64:
        body["inputImage"] = image_b64
    if dims is not None:
        body["embeddingConfig"] = {"outputEmbeddingLength": dims}
    return body


def parse_response(result: dict) -> list[float]:
    return result["embedding"]


MODEL_PREFIX = "titan-embed-image"
