from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import requests

LLM_PROVIDERS = ("openai", "gemini", "anthropic", "deepseek")
IMAGE_PROVIDERS = ("disabled", "openai", "gemini")

OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OPENAI_FALLBACK_IMAGE_MODELS = ("gpt-image-1", "dall-e-3", "dall-e-2")
GEMINI_FALLBACK_IMAGE_MODELS = (
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "imagen-4.0-generate-001",
)


class ProviderError(RuntimeError):
    pass


@dataclass
class ImageAttachment:
    mime_type: str
    data: bytes
    caption: str = ""


@dataclass
class ProviderConfig:
    provider: str
    api_key: str
    model: str
    base_url: str | None = None

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.provider == "openai":
            return OPENAI_BASE_URL
        if self.provider == "gemini":
            return GEMINI_BASE_URL
        if self.provider == "anthropic":
            return ANTHROPIC_BASE_URL
        if self.provider == "deepseek":
            return DEEPSEEK_BASE_URL
        raise ProviderError(f"Unsupported provider: {self.provider}")


def supports_image_generation(provider: str) -> bool:
    return provider in {"openai", "gemini"}


def supports_vision_review(provider: str) -> bool:
    return provider in {"openai", "gemini", "anthropic"}


def discover_models(
    provider: str,
    api_key: str,
    *,
    purpose: str = "text",
    base_url: str | None = None,
    timeout: int = 45,
) -> list[str]:
    provider = provider.lower().strip()
    if purpose not in {"text", "image"}:
        raise ValueError(f"Unsupported model purpose: {purpose}")
    if not api_key.strip():
        raise ProviderError("Enter an API key before loading models.")

    if provider == "openai":
        models = _list_openai_models(api_key, base_url=base_url, timeout=timeout)
        if purpose == "text":
            return _filter_openai_text_models(models)
        return _filter_openai_image_models(models)

    if provider == "gemini":
        models = _list_gemini_models(api_key, timeout=timeout)
        if purpose == "text":
            return _filter_gemini_text_models(models)
        return _filter_gemini_image_models(models)

    if provider == "anthropic":
        if purpose == "image":
            return []
        return _list_anthropic_models(api_key, timeout=timeout)

    if provider == "deepseek":
        if purpose == "image":
            return []
        return _list_deepseek_models(api_key, base_url=base_url, timeout=timeout)

    raise ProviderError(f"Unsupported provider: {provider}")


def generate_text(
    config: ProviderConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    attachments: list[ImageAttachment] | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    provider = config.provider.lower().strip()
    if provider == "openai":
        return _generate_openai_text(
            config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attachments=attachments or [],
            json_mode=json_mode,
            temperature=temperature,
            timeout=timeout,
        )
    if provider == "gemini":
        return _generate_gemini_text(
            config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attachments=attachments or [],
            json_mode=json_mode,
            temperature=temperature,
            timeout=timeout,
        )
    if provider == "anthropic":
        return _generate_anthropic_text(
            config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attachments=attachments or [],
            json_mode=json_mode,
            temperature=temperature,
            timeout=timeout,
        )
    if provider == "deepseek":
        return _generate_deepseek_text(
            config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=json_mode,
            temperature=temperature,
            timeout=timeout,
        )
    raise ProviderError(f"Unsupported provider: {config.provider}")


def generate_image(
    config: ProviderConfig,
    *,
    prompt: str,
    timeout: int = 180,
) -> bytes:
    provider = config.provider.lower().strip()
    if provider == "openai":
        return _generate_openai_image(config, prompt=prompt, timeout=timeout)
    if provider == "gemini":
        return _generate_gemini_image(config, prompt=prompt, timeout=timeout)
    raise ProviderError(f"Provider does not support image generation: {config.provider}")


def _extract_response_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason

    if isinstance(payload, dict):
        if "error" in payload:
            error = payload["error"]
            if isinstance(error, dict):
                message = error.get("message") or error.get("type")
                if message:
                    return str(message)
            return str(error)
        if "message" in payload:
            return str(payload["message"])
    return response.text.strip() or response.reason


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        json=json_body,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise ProviderError(f"{response.status_code} from {url}: {_extract_response_error(response)}")
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(f"Invalid JSON response from {url}.") from exc


def _list_openai_models(api_key: str, *, base_url: str | None, timeout: int) -> list[dict[str, Any]]:
    payload = _request_json(
        "GET",
        f"{(base_url or OPENAI_BASE_URL).rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        timeout=timeout,
    )
    return list(payload.get("data", []))


def _filter_openai_text_models(models: list[dict[str, Any]]) -> list[str]:
    excluded = ("audio", "embedding", "image", "moderation", "realtime", "search", "transcribe", "tts", "whisper")
    model_ids = []
    for entry in models:
        model_id = str(entry.get("id", "")).strip()
        if not model_id:
            continue
        if any(token in model_id for token in excluded):
            continue
        model_ids.append(model_id)
    return sorted(set(model_ids))


def _filter_openai_image_models(models: list[dict[str, Any]]) -> list[str]:
    discovered = {
        str(entry.get("id", "")).strip()
        for entry in models
        if "image" in str(entry.get("id", "")).lower() or str(entry.get("id", "")).startswith("dall-e")
    }
    if not discovered:
        discovered = set(OPENAI_FALLBACK_IMAGE_MODELS)
    return sorted(discovered)


def _list_gemini_models(api_key: str, *, timeout: int) -> list[dict[str, Any]]:
    payload = _request_json(
        "GET",
        f"{GEMINI_BASE_URL}/models",
        params={"key": api_key.strip()},
        timeout=timeout,
    )
    return list(payload.get("models", []))


def _filter_gemini_text_models(models: list[dict[str, Any]]) -> list[str]:
    excluded = ("embed", "imagen", "image", "aqa", "tts", "speech")
    model_ids = []
    for entry in models:
        methods = entry.get("supportedGenerationMethods", []) or []
        if "generateContent" not in methods:
            continue
        model_id = _strip_gemini_model_prefix(str(entry.get("name", "")))
        if not model_id or any(token in model_id for token in excluded):
            continue
        model_ids.append(model_id)
    return sorted(set(model_ids))


def _filter_gemini_image_models(models: list[dict[str, Any]]) -> list[str]:
    model_ids = {
        _strip_gemini_model_prefix(str(entry.get("name", "")))
        for entry in models
        if any(token in _strip_gemini_model_prefix(str(entry.get("name", ""))) for token in ("image", "imagen"))
    }
    model_ids = {item for item in model_ids if item}
    if not model_ids:
        model_ids = set(GEMINI_FALLBACK_IMAGE_MODELS)
    return sorted(model_ids)


def _list_anthropic_models(api_key: str, *, timeout: int) -> list[str]:
    payload = _request_json(
        "GET",
        f"{ANTHROPIC_BASE_URL}/v1/models",
        headers={
            "x-api-key": api_key.strip(),
            "anthropic-version": "2023-06-01",
        },
        timeout=timeout,
    )
    return sorted({str(entry.get("id", "")).strip() for entry in payload.get("data", []) if entry.get("id")})


def _list_deepseek_models(api_key: str, *, base_url: str | None, timeout: int) -> list[str]:
    payload = _request_json(
        "GET",
        f"{(base_url or DEEPSEEK_BASE_URL).rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        timeout=timeout,
    )
    return sorted({str(entry.get("id", "")).strip() for entry in payload.get("data", []) if entry.get("id")})


def _generate_openai_text(
    config: ProviderConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    attachments: list[ImageAttachment],
    json_mode: bool,
    temperature: float,
    timeout: int,
) -> str:
    content: str | list[dict[str, Any]]
    if attachments:
        content = [{"type": "text", "text": user_prompt}]
        for item in attachments:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
                    },
                }
            )
    else:
        content = user_prompt

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = _request_json(
        "POST",
        f"{config.resolved_base_url()}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key.strip()}"},
        json_body=payload,
        timeout=timeout,
    )
    try:
        content_value = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("OpenAI returned an unexpected response shape.") from exc

    if isinstance(content_value, list):
        return "\n".join(item.get("text", "") for item in content_value if isinstance(item, dict)).strip()
    return str(content_value).strip()


def _generate_deepseek_text(
    config: ProviderConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool,
    temperature: float,
    timeout: int,
) -> str:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = _request_json(
        "POST",
        f"{config.resolved_base_url()}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key.strip()}"},
        json_body=payload,
        timeout=timeout,
    )
    try:
        return str(response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("DeepSeek returned an unexpected response shape.") from exc


def _generate_anthropic_text(
    config: ProviderConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    attachments: list[ImageAttachment],
    json_mode: bool,
    temperature: float,
    timeout: int,
) -> str:
    if json_mode:
        user_prompt = (
            "Return valid JSON only. Do not use markdown fences.\n\n"
            f"{user_prompt}"
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for item in attachments:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.mime_type,
                    "data": base64.b64encode(item.data).decode("ascii"),
                },
            }
        )

    payload = {
        "model": config.model,
        "max_tokens": 4096,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
    }
    response = _request_json(
        "POST",
        f"{config.resolved_base_url()}/v1/messages",
        headers={
            "x-api-key": config.api_key.strip(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json_body=payload,
        timeout=timeout,
    )
    chunks = []
    for item in response.get("content", []):
        if item.get("type") == "text" and item.get("text"):
            chunks.append(item["text"])
    return "\n".join(chunks).strip()


def _generate_gemini_text(
    config: ProviderConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    attachments: list[ImageAttachment],
    json_mode: bool,
    temperature: float,
    timeout: int,
) -> str:
    parts: list[dict[str, Any]] = [{"text": user_prompt}]
    for item in attachments:
        parts.append(
            {
                "inline_data": {
                    "mime_type": item.mime_type,
                    "data": base64.b64encode(item.data).decode("ascii"),
                }
            }
        )

    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature},
    }
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    response = _request_json(
        "POST",
        f"{GEMINI_BASE_URL}/models/{_strip_gemini_model_prefix(config.model)}:generateContent",
        params={"key": config.api_key.strip()},
        json_body=payload,
        timeout=timeout,
    )
    return _extract_gemini_text(response)


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    chunks = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                chunks.append(part["text"])
    text = "\n".join(chunks).strip()
    if not text:
        raise ProviderError("Gemini returned no text content.")
    return text


def _generate_openai_image(config: ProviderConfig, *, prompt: str, timeout: int) -> bytes:
    payload = {
        "model": config.model,
        "prompt": prompt,
        "response_format": "b64_json",
    }
    response = _request_json(
        "POST",
        f"{config.resolved_base_url()}/images/generations",
        headers={"Authorization": f"Bearer {config.api_key.strip()}"},
        json_body=payload,
        timeout=timeout,
    )
    try:
        item = response["data"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("OpenAI returned an unexpected image response shape.") from exc

    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])

    if item.get("url"):
        image_response = requests.get(item["url"], timeout=timeout)
        image_response.raise_for_status()
        return image_response.content

    raise ProviderError("OpenAI image generation returned no usable image data.")


def _generate_gemini_image(config: ProviderConfig, *, prompt: str, timeout: int) -> bytes:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    response = _request_json(
        "POST",
        f"{GEMINI_BASE_URL}/models/{_strip_gemini_model_prefix(config.model)}:generateContent",
        params={"key": config.api_key.strip()},
        json_body=payload,
        timeout=timeout,
    )

    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                return base64.b64decode(inline_data["data"])
    raise ProviderError("Gemini image generation returned no inline image bytes.")


def _strip_gemini_model_prefix(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name
