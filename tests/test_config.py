"""Provider configuration checks without reading the user's .env file."""

import os
import unittest
from unittest.mock import patch

import config
from config import load_settings


class ConfigTests(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_openrouter_remains_default(self, _dotenv):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "router-key", "OPENROUTER_MODEL": "router/model",
        }, clear=True):
            settings = load_settings()
        self.assertEqual(settings.provider, "openrouter")
        self.assertEqual(settings.api_key, "router-key")
        self.assertEqual(settings.model, "router/model")
        self.assertEqual(settings.base_url, "https://openrouter.ai/api/v1")

    @patch("config.load_dotenv")
    def test_nvidia_needs_only_nvidia_credentials(self, _dotenv):
        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "nvidia-key", "NVIDIA_MODEL": "vendor/tool-model",
            "GOOGLE_PLACES_API_KEY": "places-key",
        }, clear=True):
            settings = load_settings("nvidia")
        self.assertEqual(settings.provider, "nvidia")
        self.assertEqual(settings.api_key, "nvidia-key")
        self.assertEqual(settings.model, "vendor/tool-model")
        self.assertEqual(settings.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertEqual(settings.places_api_key, "places-key")

    @patch("config.load_dotenv")
    def test_cloudflare_account_endpoint_and_missing_values(self, _dotenv):
        with patch.dict(os.environ, {
            "CLOUDFLARE_API_TOKEN": "cf-token",
            "CLOUDFLARE_MODEL": "@cf/vendor/tool-model",
            "CLOUDFLARE_ACCOUNT_ID": "account123",
        }, clear=True):
            settings = load_settings("cloudflare")
        self.assertEqual(settings.provider, "cloudflare")
        self.assertEqual(settings.api_key, "cf-token")
        self.assertEqual(settings.model, "@cf/vendor/tool-model")
        self.assertEqual(
            settings.base_url,
            "https://api.cloudflare.com/client/v4/accounts/account123/ai/v1",
        )
        with patch.dict(os.environ, {
            "CLOUDFLARE_API_TOKEN": "cf-token", "CLOUDFLARE_MODEL": "@cf/model",
        }, clear=True):
            with self.assertRaisesRegex(config.ConfigurationError, "CLOUDFLARE_ACCOUNT_ID"):
                load_settings("cloudflare")

    @patch("config.load_dotenv")
    def test_groq_and_gemini_endpoints_and_credentials(self, _dotenv):
        for provider, key_name, model_name, endpoint in (
            ("groq", "GROQ_API_KEY", "GROQ_MODEL", "https://api.groq.com/openai/v1"),
            ("gemini", "GEMINI_API_KEY", "GEMINI_MODEL",
             "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ):
            with self.subTest(provider=provider), patch.dict(os.environ, {
                key_name: "test-key", model_name: "tool-model",
            }, clear=True):
                settings = load_settings(provider)
                self.assertEqual(settings.provider, provider)
                self.assertEqual(settings.api_key, "test-key")
                self.assertEqual(settings.model, "tool-model")
                self.assertEqual(settings.base_url, endpoint)
            with self.subTest(provider=provider, case="missing"), patch.dict(
                os.environ, {}, clear=True
            ):
                with self.assertRaisesRegex(config.ConfigurationError, key_name):
                    load_settings(provider)

    @patch("config.load_dotenv")
    def test_unknown_and_unconfigured_provider(self, _dotenv):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(config.ConfigurationError, "Unknown LLM provider"):
                load_settings("unknown")
            with self.assertRaisesRegex(config.ConfigurationError, "NVIDIA_API_KEY, NVIDIA_MODEL"):
                load_settings("nvidia")


if __name__ == "__main__":
    unittest.main()
