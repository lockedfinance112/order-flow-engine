import config
import ai_providers


AVAILABLE_PROVIDERS = ["openai", "gemini", "claude", "deepseek"]

DEFAULT_MODELS = {
    "openai": config.OPENAI_MODEL,
    "gemini": config.GEMINI_MODEL,
    "claude": config.CLAUDE_MODEL,
    "deepseek": config.DEEPSEEK_MODEL,
}

ENV_API_KEYS = {
    "openai": config.OPENAI_API_KEY,
    "gemini": config.GEMINI_API_KEY,
    "claude": config.ANTHROPIC_API_KEY,
    "deepseek": config.DEEPSEEK_API_KEY,
}


class RuntimeAIConfig:
    """Memory-only AI settings for the current engine process."""

    def __init__(self):
        provider = (config.AI_PROVIDER or "openai").lower().strip()
        if provider not in AVAILABLE_PROVIDERS:
            provider = "openai"
        self.provider = provider
        self.model = DEFAULT_MODELS.get(provider, "")
        self.enabled = bool(config.AI_ENABLED)
        self._api_keys = {}

    def update(self, provider=None, model=None, api_key=None, enabled=None):
        if provider is not None:
            next_provider = str(provider).lower().strip()
            if next_provider not in AVAILABLE_PROVIDERS:
                raise ValueError(f"Unsupported AI provider '{provider}'.")
            if next_provider != self.provider:
                self.provider = next_provider
                self.model = DEFAULT_MODELS.get(next_provider, "")

        if model is not None:
            next_model = str(model).strip()
            if not next_model:
                raise ValueError("Model cannot be empty.")
            self.model = next_model

        if api_key is not None:
            next_key = str(api_key).strip()
            if next_key:
                self._api_keys[self.provider] = next_key
                ai_providers.register_runtime_secret(next_key)

        if enabled is not None:
            self.enabled = bool(enabled)

    def get_api_key(self, provider=None):
        provider_name = (provider or self.provider).lower().strip()
        return self._api_keys.get(provider_name) or ENV_API_KEYS.get(provider_name, "")

    def get_model(self, provider=None):
        provider_name = (provider or self.provider).lower().strip()
        if provider_name == self.provider:
            return self.model
        return DEFAULT_MODELS.get(provider_name, "")

    def is_configured(self, provider=None):
        key = self.get_api_key(provider)
        return bool(key and len(key.strip()) > 5)

    def to_public_dict(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "configured": self.is_configured(),
            "enabled": self.enabled,
            "available_providers": AVAILABLE_PROVIDERS,
            "default_models": DEFAULT_MODELS,
        }
