"""Application settings loaded from the local environment."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """A required application setting is missing."""


PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
    "nvidia": "NVIDIA",
    "cloudflare": "Cloudflare Workers AI",
    "groq": "Groq",
    "gemini": "Gemini",
}

PROVIDER_ENV = {
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"),
    "nvidia": ("NVIDIA_API_KEY", "NVIDIA_MODEL"),
    "cloudflare": ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_MODEL"),
    "groq": ("GROQ_API_KEY", "GROQ_MODEL"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL"),
}


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    calendar_id: str = ""
    calendar_api_key: str = ""
    places_api_key: str = ""
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"


def load_settings(provider: str = "openrouter") -> Settings:
    # Resolve .env relative to this file so launching from another directory works.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    if provider not in PROVIDER_ENV:
        raise ConfigurationError(f"Unknown LLM provider: {provider}.")
    key_name, model_name = PROVIDER_ENV[provider]
    api_key = os.getenv(key_name, "").strip()
    model = os.getenv(model_name, "").strip()
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip() if provider == "cloudflare" else ""

    required = [(key_name, api_key), (model_name, model)]
    if provider == "cloudflare":
        required.append(("CLOUDFLARE_ACCOUNT_ID", account_id))
    missing = [name for name, value in required if not value]
    if missing:
        raise ConfigurationError(
            f"Missing {', '.join(missing)}. Set them in chat-app/.env and rerun the app."
        )

    base_url = {
        "openrouter": "https://openrouter.ai/api/v1",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "cloudflare": (
            f"https://api.cloudflare.com/client/v4/accounts/{quote(account_id, safe='')}/ai/v1"
        ),
        "groq": "https://api.groq.com/openai/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }[provider]

    return Settings(
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=base_url,
        calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "").strip(),
        calendar_api_key=os.getenv("GOOGLE_CALENDAR_API_KEY", "").strip(),
        places_api_key=os.getenv("GOOGLE_PLACES_API_KEY", "").strip(),
    )
