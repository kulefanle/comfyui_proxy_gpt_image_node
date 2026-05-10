"""
ComfyUI custom node: Proxy GPT Image Generator

This node is designed for OpenAI-compatible proxy services that expose either
`/v1/images/generations` or `/v1/chat/completions` for image generation.

Risk notes:
- API keys are accepted as node inputs, but they are never printed, returned, or
  written to disk by this node. Do not share workflow JSON files that contain
  real keys, because ComfyUI stores widget values inside exported workflows.
- The node downloads image URLs returned by the proxy. Only use a proxy/base URL
  you trust, because remote image URLs can reveal your server IP and consume
  bandwidth.
- If no valid image is found, the node raises an error instead of returning a
  blank placeholder. This makes authorization, parsing, or proxy errors visible.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import requests
import torch
from PIL import Image


IMAGE_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:png|jpg|jpeg|webp)(?:\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)
DATA_URL_RE = re.compile(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)")
BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]{200,}$")


def _join_endpoint(base_url: str, route: str) -> str:
    """Build a stable endpoint without silently producing duplicate slashes."""
    clean_base = base_url.strip().rstrip("/")
    clean_route = route.strip()
    if not clean_base:
        raise ValueError("Base URL is empty.")
    if clean_route.startswith("http://") or clean_route.startswith("https://"):
        return clean_route
    return urljoin(clean_base + "/", clean_route.lstrip("/"))


def _image_to_data_url(image: torch.Tensor, fmt: str = "PNG") -> str:
    """
    Convert a ComfyUI IMAGE tensor to a data URL for chat/completions proxies.

    ComfyUI images are float tensors in [B, H, W, C]. The proxy usually accepts
    a standard data URL in an OpenAI-style `image_url` content part.
    """
    if image is None:
        raise ValueError("Reference image is missing.")

    image_np = image[0].detach().cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np).convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _bytes_to_comfy_image(image_bytes: bytes) -> torch.Tensor:
    """
    Decode image bytes into a ComfyUI IMAGE tensor.

    RGBA images are composited onto a white background. This avoids the common
    "transparent PNG looks like a white/blank image" trap while still returning
    a normal RGB tensor to PreviewImage.
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode == "RGBA":
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        pil_image = Image.alpha_composite(background, pil_image).convert("RGB")
    else:
        pil_image = pil_image.convert("RGB")

    image_np = np.asarray(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np)[None,]


def _decode_data_url(value: str) -> bytes | None:
    """Decode a `data:image/...;base64,...` string if the value contains one."""
    match = DATA_URL_RE.search(value)
    if not match:
        return None
    return base64.b64decode(match.group(1).replace("\n", "").replace(" ", ""))


def _decode_probable_base64(value: str) -> bytes | None:
    """
    Decode long raw base64 strings.

    The length and alphabet checks reduce the risk of treating normal text such
    as `auto` or an error message as an image.
    """
    compact = value.strip().replace("\n", "").replace(" ", "")
    if len(compact) < 200 or not BASE64_RE.match(compact):
        return None
    try:
        decoded = base64.b64decode(compact, validate=True)
    except Exception:
        return None
    if decoded[:8] == b"\x89PNG\r\n\x1a\n" or decoded[:3] == b"\xff\xd8\xff" or decoded[:4] == b"RIFF":
        return decoded
    return None


def _iter_strings(value: Any) -> Iterable[str]:
    """Walk a JSON-like object and yield candidate strings from common fields."""
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
        return

    if isinstance(value, dict):
        priority_keys = (
            "b64_json",
            "url",
            "image_url",
            "image",
            "file_url",
            "content",
            "text",
            "output",
            "data",
            "images",
        )
        for key in priority_keys:
            if key in value:
                item = value[key]
                if isinstance(item, dict) and "url" in item:
                    yield from _iter_strings(item["url"])
                else:
                    yield from _iter_strings(item)
        for key, item in value.items():
            if key not in priority_keys:
                yield from _iter_strings(item)


def _extract_first_image_candidate(payload: Any) -> tuple[str, bytes | None]:
    """
    Find the first usable image from a proxy response.

    Supported forms:
    - OpenAI Images API: data[0].b64_json or data[0].url
    - Chat proxies: message content containing a data URL, raw base64, markdown
      image URL, or direct HTTP image URL
    - Custom proxies: top-level `images`, `image`, `url`, or `file_url`
    """
    for text in _iter_strings(payload):
        data_url_bytes = _decode_data_url(text)
        if data_url_bytes:
            return ("data_url", data_url_bytes)

        raw_base64_bytes = _decode_probable_base64(text)
        if raw_base64_bytes:
            return ("base64", raw_base64_bytes)

        url_match = IMAGE_URL_RE.search(text)
        if url_match:
            return (url_match.group(0), None)

        stripped = text.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            return (stripped, None)

    return ("", None)


def _summarize_payload(payload: Any, limit: int = 1600) -> str:
    """
    Return a compact response summary for PreviewAny.

    This intentionally redacts large base64-like strings so the UI stays usable
    and exported workflows do not become huge because of response text.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(payload)
    text = re.sub(r"([A-Za-z0-9+/=]{180,})", "[base64 omitted]", text)
    return text[:limit]


class ProxyGPTImageGenerator:
    """ComfyUI node that calls an OpenAI-compatible image proxy."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "password": True,
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://greenapi.ink",
                        "multiline": False,
                    },
                ),
                "api_route": (
                    [
                        "/v1/images/generations",
                        "/v1/chat/completions",
                    ],
                    {
                        "default": "/v1/images/generations",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gpt-image-2",
                        "multiline": False,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "a red apple on a wooden table, realistic photo",
                        "multiline": True,
                    },
                ),
                "size": (
                    [
                        "1024x1024",
                        "1024x1536",
                        "1536x1024",
                        "auto",
                    ],
                    {
                        "default": "1024x1024",
                    },
                ),
                "quality": (
                    [
                        "low",
                        "medium",
                        "high",
                        "auto",
                    ],
                    {
                        "default": "medium",
                    },
                ),
                "background": (
                    [
                        "opaque",
                        "transparent",
                        "auto",
                    ],
                    {
                        "default": "opaque",
                    },
                ),
                "output_format": (
                    [
                        "png",
                        "jpeg",
                        "webp",
                    ],
                    {
                        "default": "png",
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 180,
                        "min": 10,
                        "max": 900,
                        "step": 10,
                    },
                ),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_source", "response_summary")
    FUNCTION = "generate"
    CATEGORY = "api/proxy"

    def generate(
        self,
        api_key: str,
        base_url: str,
        api_route: str,
        model: str,
        prompt: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        timeout_seconds: int,
        reference_image_1: torch.Tensor | None = None,
        reference_image_2: torch.Tensor | None = None,
        reference_image_3: torch.Tensor | None = None,
        reference_image_4: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str, str]:
        clean_key = api_key.strip()
        clean_prompt = prompt.strip()

        if not clean_key:
            raise ValueError("API key is empty. Fill a valid key from the same proxy service as Base URL.")
        if not clean_prompt:
            raise ValueError("Prompt is empty. The proxy may return a blank result when prompt is empty.")

        endpoint = _join_endpoint(base_url, api_route)
        headers = {
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, image/*",
        }

        references = [
            image
            for image in (reference_image_1, reference_image_2, reference_image_3, reference_image_4)
            if image is not None
        ]

        if api_route.endswith("/chat/completions"):
            # Many proxy services map image generation models onto the chat API.
            # The content-array format keeps text and optional references explicit.
            content: list[dict[str, Any]] = [{"type": "text", "text": clean_prompt}]
            for image in references:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(image)},
                    }
                )

            payload: dict[str, Any] = {
                "model": model.strip(),
                "messages": [{"role": "user", "content": content}],
                "size": size,
                "quality": quality,
                "background": background,
                "output_format": output_format,
            }
        else:
            # OpenAI Images-style JSON body. Some proxies also accept `images`
            # as data URLs for image-to-image; unsupported proxies should return
            # a clear 4xx error rather than a hidden blank image.
            payload = {
                "model": model.strip(),
                "prompt": clean_prompt,
                "n": 1,
                "size": size,
                "quality": quality,
                "background": background,
                "output_format": output_format,
            }
            if references:
                payload["images"] = [_image_to_data_url(image) for image in references]

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=int(timeout_seconds),
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to connect to proxy endpoint: {endpoint}. Detail: {exc}") from exc

        content_type = response.headers.get("content-type", "").lower()

        if not response.ok:
            # Do not include headers or API key in this error. The response body
            # is useful for diagnosing 401/403/429/model-not-found issues.
            body = response.text[:1200]
            raise RuntimeError(f"Proxy returned HTTP {response.status_code} for {endpoint}: {body}")

        if content_type.startswith("image/"):
            image_tensor = _bytes_to_comfy_image(response.content)
            return (image_tensor, endpoint, "Proxy returned image bytes directly.")

        try:
            payload_response: Any = response.json()
        except ValueError as exc:
            body = response.text[:1200]
            raise RuntimeError(f"Proxy did not return JSON or image bytes. Body: {body}") from exc

        image_source, inline_bytes = _extract_first_image_candidate(payload_response)
        summary = _summarize_payload(payload_response)

        if inline_bytes:
            image_tensor = _bytes_to_comfy_image(inline_bytes)
            return (image_tensor, image_source, summary)

        if image_source.startswith("http://") or image_source.startswith("https://"):
            try:
                image_response = requests.get(image_source, timeout=int(timeout_seconds))
                image_response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"Image URL was returned but could not be downloaded: {image_source}") from exc

            image_tensor = _bytes_to_comfy_image(image_response.content)
            return (image_tensor, image_source, summary)

        raise RuntimeError(
            "Proxy response did not contain a usable image URL, data URL, or b64_json. "
            f"Response summary: {summary}"
        )


NODE_CLASS_MAPPINGS = {
    "ProxyGPTImageGenerator": ProxyGPTImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProxyGPTImageGenerator": "Proxy GPT Image Generator",
}
