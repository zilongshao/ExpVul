import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


_state = threading.local()
MAX_RETRIES = 2
RETRY_DELAY = 1.0


def _token_state() -> Dict[str, int]:
    value = getattr(_state, "value", None)
    if value is None:
        value = {"total": 0, "prompt": 0, "completion": 0}
        _state.value = value
    return value


def reset_token_counter() -> None:
    _state.value = {"total": 0, "prompt": 0, "completion": 0}


def get_token_count() -> int:
    return int(_token_state()["total"])


def get_token_breakdown() -> Dict[str, int]:
    return dict(_token_state())


def _record_usage(payload: Dict[str, Any]) -> None:
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total = int(usage.get("total_tokens", prompt + completion) or 0)
    state = _token_state()
    state["prompt"] += prompt
    state["completion"] += completion
    state["total"] += total


def _settings() -> Tuple[str, str, str]:
    base_url = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    credential = os.environ.get("LLM_AUTH", "").strip()
    model_name = os.environ.get("LLM_MODEL", "").strip()
    return base_url, credential, model_name


def configured() -> bool:
    base_url, credential, model_name = _settings()
    return bool(base_url and credential and model_name)


def current_model() -> str:
    return _settings()[2]


def set_model(model_name: str) -> None:
    if model_name:
        os.environ["LLM_MODEL"] = model_name


def _endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("response did not contain a completion")
    message = choices[0].get("message") or {}
    value = message.get("content", "")
    if isinstance(value, list):
        value = "".join(str(part.get("text", "")) for part in value if isinstance(part, dict))
    if not isinstance(value, str):
        raise LLMError("completion content was not text")
    return value.strip()


def chat_completion(
    prompt: str,
    *,
    timeout: int = 1200,
    model: Optional[str] = None,
    temperature: Optional[float] = 0.0,
) -> str:
    base_url, credential, configured_model = _settings()
    selected_model = (model or configured_model).strip()
    if not base_url or not credential or not selected_model:
        raise LLMConfigurationError("LLM_BASE_URL, LLM_AUTH, and LLM_MODEL are required")
    import requests

    payload: Dict[str, Any] = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0 if temperature is None else float(temperature),
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {credential}"}
    error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                _endpoint(base_url),
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code < 200 or response.status_code >= 300:
                error = LLMError(f"LLM request failed with status {response.status_code}")
            else:
                body = response.json()
                _record_usage(body)
                return _content(body)
        except (requests.RequestException, ValueError, TypeError, KeyError, LLMError) as exc:
            error = exc
        if attempt + 1 < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (attempt + 1))
    if isinstance(error, LLMError):
        raise error
    raise LLMError("LLM request failed") from error


def call_with_parser(
    prompt: str,
    parser: Callable[[str], Any],
    *,
    model: Optional[str] = None,
) -> Any:
    text = chat_completion(prompt, model=model, temperature=0.0)
    return parser(text)
