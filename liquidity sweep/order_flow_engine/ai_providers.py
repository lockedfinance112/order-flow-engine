# Order Flow Engine V2.5 - AI Provider Adapters

import os
import urllib.request
import urllib.error
import time
import json
import logging
import asyncio
from typing import Dict, Any, Tuple, Optional
import re
import ssl

logger = logging.getLogger("OrderFlow.AIProviders")

_SENSITIVE_ENV_NAMES = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
)

_runtime_secret_values = set()


def register_runtime_secret(secret: Optional[str]):
    if secret and len(str(secret).strip()) >= 5:
        _runtime_secret_values.add(str(secret).strip())


def _configured_secret_values() -> list:
    secrets = []
    for name in _SENSITIVE_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if len(value) >= 5:
            secrets.append(value)
    secrets.extend(_runtime_secret_values)
    return secrets


def redact_sensitive(value: Optional[str]) -> Optional[str]:
    """Redacts provider API keys from errors, raw responses, and logs."""
    if value is None:
        return None

    text = str(value)
    for secret in _configured_secret_values():
        text = text.replace(secret, "[REDACTED]")

    patterns = [
        (r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?(?:bearer\s+)?)[^'\"\s,}]+", r"\1[REDACTED]"),
        (r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1[REDACTED]"),
        (r"(?i)([?&]key=)[^&\s'\"]+", r"\1[REDACTED]"),
        (r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{8,}", "[REDACTED]"),
        (r"AIzaSy[A-Za-z0-9_\-]{10,}", "[REDACTED]"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def clean_json_text(text: str) -> str:
    """Strips markdown wrappers (like ```json ... ```) from response text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def _make_http_post(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: int, ssl_verify: bool = True) -> Tuple[bool, int, Optional[str], Optional[str]]:
    """Helper to perform synchronous blocking HTTP POST request using urllib."""
    start_time = time.time()
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        context = None if ssl_verify else ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            resp_body = response.read().decode("utf-8")
            latency = int((time.time() - start_time) * 1000)
            return True, latency, resp_body, None
    except urllib.error.HTTPError as e:
        latency = int((time.time() - start_time) * 1000)
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        return False, latency, redact_sensitive(err_body), redact_sensitive(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        latency = int((time.time() - start_time) * 1000)
        return False, latency, None, redact_sensitive(f"URL Error: {e.reason}")
    except Exception as e:
        latency = int((time.time() - start_time) * 1000)
        return False, latency, None, redact_sensitive(f"Exception: {str(e)}")

class AIProvider:
    def __init__(self, provider_name: str, api_key: str, model: str, timeout: int = 20, ssl_verify: bool = True):
        self.provider_name = provider_name
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.ssl_verify = ssl_verify

    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validates that the provider settings and API key are configured correctly."""
        if not self.api_key or len(self.api_key.strip()) < 5:
            return False, f"Missing or invalid API key for provider '{self.provider_name}'."
        if not self.model:
            return False, f"Missing model configuration for provider '{self.provider_name}'."
        return True, None

    async def interpret(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Wrapper to run the blocking request in a thread pool without blocking the asyncio loop."""
        valid, err_msg = self.validate_config()
        if not valid:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": None,
                "latency_ms": 0,
                "error": redact_sensitive(err_msg)
            }
        
        # Execute the HTTP request in a thread pool
        return await asyncio.to_thread(self._execute_request, system_prompt, user_prompt)

    def _execute_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _execute_request.")


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str, timeout: int = 20, ssl_verify: bool = True):
        super().__init__("openai", api_key, model, timeout, ssl_verify)

    def _execute_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        ok, latency, resp_body, err = _make_http_post(url, headers, body, self.timeout, self.ssl_verify)
        if not ok or not resp_body:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(err or "Empty response")
            }

        try:
            resp_json = json.loads(resp_body)
            raw_text = clean_json_text(resp_json["choices"][0]["message"]["content"])
            parsed_json = json.loads(raw_text)
            return {
                "ok": True,
                "provider": self.provider_name,
                "model": self.model,
                "json": parsed_json,
                "raw_text": raw_text,
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(f"JSON Parse Error: {str(e)}")
            }


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str, timeout: int = 20, ssl_verify: bool = True):
        super().__init__("gemini", api_key, model, timeout, ssl_verify)

    def _execute_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        ok, latency, resp_body, err = _make_http_post(url, headers, body, self.timeout, self.ssl_verify)
        if not ok or not resp_body:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(err or "Empty response")
            }

        try:
            resp_json = json.loads(resp_body)
            raw_text = clean_json_text(resp_json["candidates"][0]["content"]["parts"][0]["text"])
            parsed_json = json.loads(raw_text)
            return {
                "ok": True,
                "provider": self.provider_name,
                "model": self.model,
                "json": parsed_json,
                "raw_text": raw_text,
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(f"Gemini JSON Parse Error: {str(e)}")
            }


class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str, model: str, timeout: int = 20, ssl_verify: bool = True):
        super().__init__("claude", api_key, model, timeout, ssl_verify)

    def _execute_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        body = {
            "model": self.model,
            "max_tokens": 4000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }

        ok, latency, resp_body, err = _make_http_post(url, headers, body, self.timeout, self.ssl_verify)
        if not ok or not resp_body:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(err or "Empty response")
            }

        try:
            resp_json = json.loads(resp_body)
            raw_text = clean_json_text(resp_json["content"][0]["text"])
            parsed_json = json.loads(raw_text)
            return {
                "ok": True,
                "provider": self.provider_name,
                "model": self.model,
                "json": parsed_json,
                "raw_text": raw_text,
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(f"Claude JSON Parse Error: {str(e)}")
            }


class DeepSeekProvider(AIProvider):
    def __init__(self, api_key: str, model: str, timeout: int = 20, ssl_verify: bool = True):
        super().__init__("deepseek", api_key, model, timeout, ssl_verify)

    def _execute_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        ok, latency, resp_body, err = _make_http_post(url, headers, body, self.timeout, self.ssl_verify)
        if not ok or not resp_body:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(err or "Empty response")
            }

        try:
            resp_json = json.loads(resp_body)
            raw_text = clean_json_text(resp_json["choices"][0]["message"]["content"])
            parsed_json = json.loads(raw_text)
            return {
                "ok": True,
                "provider": self.provider_name,
                "model": self.model,
                "json": parsed_json,
                "raw_text": raw_text,
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            return {
                "ok": False,
                "provider": self.provider_name,
                "model": self.model,
                "json": None,
                "raw_text": redact_sensitive(resp_body),
                "latency_ms": latency,
                "error": redact_sensitive(f"DeepSeek JSON Parse Error: {str(e)}")
            }


def get_provider(provider_name: str, timeout: int = 20, runtime_config=None) -> AIProvider:
    """Factory function to build provider instances from config settings."""
    import config
    provider_name = provider_name.lower().strip()
    ssl_verify = config.AI_SSL_VERIFY

    def api_key(default_key: str) -> str:
        if runtime_config is not None:
            return runtime_config.get_api_key(provider_name)
        return default_key

    def model(default_model: str) -> str:
        if runtime_config is not None:
            return runtime_config.get_model(provider_name)
        return default_model
    
    if provider_name == "openai":
        return OpenAIProvider(api_key(config.OPENAI_API_KEY), model(config.OPENAI_MODEL), timeout, ssl_verify)
    elif provider_name == "gemini":
        return GeminiProvider(api_key(config.GEMINI_API_KEY), model(config.GEMINI_MODEL), timeout, ssl_verify)
    elif provider_name == "claude":
        return ClaudeProvider(api_key(config.ANTHROPIC_API_KEY), model(config.CLAUDE_MODEL), timeout, ssl_verify)
    elif provider_name == "deepseek":
        return DeepSeekProvider(api_key(config.DEEPSEEK_API_KEY), model(config.DEEPSEEK_MODEL), timeout, ssl_verify)
    else:
        # Fallback provider that always fails validation gracefully
        return AIProvider(provider_name, "", "", timeout)
