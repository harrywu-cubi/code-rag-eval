from __future__ import annotations
import os

# Disable extended-thinking on every request. MiMo enables thinking by default;
# we want plain completions for both generation and the judge.
THINKING_DISABLED = {"type": "disabled"}

# Default MiMo (Xiaomi) endpoint. Used whenever MIMO_API_KEY is set unless an
# explicit base_url arg or MIMO_BASE_URL env var overrides it.
MIMO_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/anthropic"


def build_anthropic(base_url: str | None = None):
    """Build an Anthropic SDK client.

    When ``MIMO_API_KEY`` is set, authenticate against the MiMo endpoint via the
    SDK's ``auth_token`` parameter, which sends ``Authorization: Bearer`` —
    MiMo does not accept ``api_key``'s ``x-api-key`` header. The endpoint
    defaults to ``MIMO_DEFAULT_BASE_URL``; the ``base_url`` argument, then the
    ``MIMO_BASE_URL`` env var, override it. So only ``MIMO_API_KEY`` is required.

    When ``MIMO_API_KEY`` is unset, fall back to the SDK defaults
    (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``).
    """
    from anthropic import Anthropic

    mimo_key = os.environ.get("MIMO_API_KEY")
    if mimo_key:
        url = base_url or os.environ.get("MIMO_BASE_URL") or MIMO_DEFAULT_BASE_URL
        return Anthropic(auth_token=mimo_key, base_url=url)
    if base_url:
        return Anthropic(base_url=base_url)
    return Anthropic()
