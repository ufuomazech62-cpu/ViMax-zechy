"""
Provider preset system for Visualiser chat model configuration.

Supports auto-detection and resolution of LLM provider settings,
allowing users to specify a provider name (e.g., ``google_vertexai``)
instead of manually configuring base_url and model details.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "google_vertexai": {
        "env_project": "GOOGLE_CLOUD_PROJECT",
        "env_location": "GOOGLE_CLOUD_LOCATION",
        "default_model": "gemini-3.1-flash",
        "models": [
            "gemini-3.1-flash",
            "gemini-3.1-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        "temperature_range": (0.0, 2.0),
    },
    "google_genai": {
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ],
        "temperature_range": (0.0, 2.0),
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "env_key": "MINIMAX_API_KEY",
        "default_model": "MiniMax-M2.7",
        "models": [
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ],
        "temperature_range": (0.0, 1.0),
    },
}


def resolve_chat_model_config(init_args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve provider presets and return final ``init_chat_model`` kwargs.

    For ``google_vertexai``: injects ``project`` and ``location`` from env vars,
    keeps ``model_provider`` as ``"google_vertexai"`` so LangChain uses
    ``ChatVertexAI``.

    For ``google_genai``: injects ``api_key`` from env var, keeps
    ``model_provider`` as ``"google_genai"`` so LangChain uses
    ``ChatGoogleGenerativeAI``.

    For ``minimax`` and other OpenAI-compatible providers: rewrites
    ``model_provider`` to ``"openai"`` and fills ``base_url``.

    For unknown providers the dict is returned unchanged.
    """
    args = dict(init_args)  # shallow copy
    provider = args.get("model_provider", "openai")

    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return args

    # --- Vertex AI: project + location (no api_key needed on Cloud Run) ---
    if provider == "google_vertexai":
        if not args.get("project"):
            env_project = preset.get("env_project", "")
            env_val = os.environ.get(env_project, "")
            if env_val:
                args["project"] = env_val
                logger.info("Using Vertex AI project from %s", env_project)
        if not args.get("location"):
            env_location = preset.get("env_location", "")
            env_val = os.environ.get(env_location, "us-central1")
            if env_val:
                args["location"] = env_val
                logger.info("Using Vertex AI location %s", env_val)
        # Remove empty strings so LangChain doesn't choke
        if not args.get("project"):
            args.pop("project", None)
        if not args.get("location"):
            args.pop("location", None)

    # --- Google AI Studio: api_key ---
    if provider == "google_genai":
        if not args.get("api_key"):
            env_key = preset.get("env_key", "")
            env_val = os.environ.get(env_key, "")
            if env_val:
                args["api_key"] = env_val
                logger.info("Using Google AI key from %s", env_key)

    # --- OpenAI-compatible: base_url + api_key ---
    if preset.get("base_url") and not args.get("base_url"):
        args["base_url"] = preset["base_url"]
    if preset.get("env_key") and provider not in ("google_vertexai", "google_genai"):
        if not args.get("api_key"):
            env_key = preset.get("env_key", "")
            env_val = os.environ.get(env_key, "")
            if env_val:
                args["api_key"] = env_val
                logger.info("Using %s API key from %s", provider, env_key)

    # default model
    if not args.get("model"):
        args["model"] = preset["default_model"]
        logger.info("Defaulting to model %s for provider %s", args["model"], provider)

    # temperature clamping
    temp_range = preset.get("temperature_range")
    if temp_range and "temperature" in args and args["temperature"] is not None:
        lo, hi = temp_range
        original = args["temperature"]
        args["temperature"] = max(lo, min(hi, original))
        if args["temperature"] != original:
            logger.warning(
                "Clamped temperature %.2f -> %.2f for provider %s",
                original, args["temperature"], provider,
            )

    # Only rewrite to openai-compatible for providers that need it.
    # Native LangChain providers are kept as-is.
    if preset.get("base_url"):
        args["model_provider"] = "openai"

    return args


def detect_provider_from_env() -> Optional[str]:
    """Return the name of a provider whose credentials are found in the environment.

    Checks ``PROVIDER_PRESETS`` in definition order and returns the first
    match, or ``None`` if no credentials are set.
    """
    for name, preset in PROVIDER_PRESETS.items():
        if name == "google_vertexai":
            env_project = preset.get("env_project", "")
            if env_project and os.environ.get(env_project):
                return name
        else:
            env_key = preset.get("env_key", "")
            if env_key and os.environ.get(env_key):
                return name
    return None
