"""Optional LLM backend for the shopping agent (OpenAI or xAI/Grok).

Both providers expose the same OpenAI-compatible ``/chat/completions`` API, so
one client serves either. Which one is used is decided by environment only -
no code change and no key in the repo.

Design rules, driven by `docs/submission_rules.md`:

* **The agent must work without it.** Official scoring may disable network
  access, so every failure path returns ``None`` and the caller falls back to
  deterministic behaviour. Nothing in this module may raise.
* **Standard library only.** No dependencies to install in the harness.
* **The key never touches the repo.** It is read from the environment or from
  a gitignored ``.env`` file.
* **Spend is bounded.** Responses are cached and call budgets cap the worst case.

Configuration
-------------
``OPENAI_API_KEY`` / ``XAI_API_KEY``  whichever is set selects the provider
``LLM_PROVIDER``     ``openai`` or ``xai`` - only needed if both keys are set
``OPENAI_MODEL`` / ``XAI_MODEL``      **required**: the exact model id
``OPENAI_BASE_URL`` / ``XAI_BASE_URL``  override the endpoint (proxies, Azure)
``AGENT_USE_LLM=0``  force the deterministic path
``AGENT_LLM_BUDGET`` max completions per process (default 4000)

Quick start
-----------
    echo 'OPENAI_API_KEY=sk-...' >> .env
    python3 -m starter.llm_client --models    # list ids your key can use
    echo 'OPENAI_MODEL=<id from above>' >> .env
    python3 -m starter.llm_client --test      # one live call, end to end
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


PROVIDERS = {
    "openai": {
        "keys": ("OPENAI_API_KEY",),
        "model": "OPENAI_MODEL",
        "base": "OPENAI_BASE_URL",
        "default_base": "https://api.openai.com/v1",
    },
    "xai": {
        "keys": ("XAI_API_KEY", "GROK_API_KEY"),
        "model": "XAI_MODEL",
        "base": "XAI_BASE_URL",
        "default_base": "https://api.x.ai/v1",
    },
}

DEFAULT_TIMEOUT = 20.0
DEFAULT_BUDGET = 4000
MAX_CONSECUTIVE_FAILURES = 3


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a gitignored .env. Existing vars win."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_provider() -> tuple[str | None, str]:
    """(provider, reason). Explicit choice wins, else whichever key is present."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            return None, f"LLM_PROVIDER={explicit!r} is not one of {sorted(PROVIDERS)}"
        if not _key_for(explicit):
            return None, f"LLM_PROVIDER={explicit} but no API key is set for it"
        return explicit, f"explicitly selected via LLM_PROVIDER={explicit}"
    found = [name for name in PROVIDERS if _key_for(name)]
    if not found:
        return None, "no API key found (set OPENAI_API_KEY or XAI_API_KEY)"
    if len(found) > 1:
        return found[0], f"multiple keys set {found}; using {found[0]} (set LLM_PROVIDER to choose)"
    return found[0], f"{found[0]} key detected"


def _key_for(provider: str) -> str:
    for name in PROVIDERS[provider]["keys"]:
        value = os.environ.get(name)
        if value:
            return value
    return ""


class LLMClient:
    """OpenAI-compatible chat client that fails soft, never hard."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        budget: int | None = None,
    ) -> None:
        self.provider = provider or resolve_provider()[0] or "openai"
        spec = PROVIDERS.get(self.provider, PROVIDERS["openai"])
        self.api_key = api_key or _key_for(self.provider)
        self.model = model or os.environ.get(spec["model"]) or ""
        self.base_url = (base_url or os.environ.get(spec["base"]) or spec["default_base"]).rstrip("/")
        self.timeout = timeout
        self.budget = budget if budget is not None else int(os.environ.get("AGENT_LLM_BUDGET", DEFAULT_BUDGET))
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.last_error: str | None = None
        self.latencies: list[float] = []
        self._consecutive_failures = 0
        self._disabled = os.environ.get("AGENT_USE_LLM", "1") == "0"
        self._cache: dict[str, str] = {}
        # Model families disagree on parameter names. Anything the endpoint
        # rejects is recorded here and dropped from later requests, so a newer
        # or older model works without a code change.
        self._unsupported: set[str] = set()

    # -- status -------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model) and not self._disabled and self.calls < self.budget

    def why_unavailable(self) -> str | None:
        if self._disabled:
            return "disabled (AGENT_USE_LLM=0, or too many consecutive failures)"
        if not self.api_key:
            return f"no API key for provider {self.provider!r}"
        if not self.model:
            spec = PROVIDERS.get(self.provider, PROVIDERS["openai"])
            return (f"no model id: set {spec['model']}. "
                    f"Run `python3 -m starter.llm_client --models` to list valid ids.")
        if self.calls >= self.budget:
            return f"call budget exhausted ({self.budget})"
        return None

    # -- requests -----------------------------------------------------------
    def _request(self, path: str, payload: dict | None = None, method: str = "POST") -> dict | None:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def list_models(self) -> list[str] | None:
        """Model ids this key can use, straight from the provider."""
        try:
            body = self._request("/models", method="GET")
        except Exception as error:
            self.last_error = _describe(error)
            return None
        data = (body or {}).get("data") or []
        return sorted(str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id"))

    def _payload(self, system: str, user: str, max_tokens: int) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if "temperature" not in self._unsupported:
            payload["temperature"] = 0          # deterministic where supported
        if "seed" not in self._unsupported:
            payload["seed"] = 7                 # best-effort determinism; not guaranteed
        if "max_tokens" in self._unsupported:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if "response_format" not in self._unsupported:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str | None:
        """Return the assistant text, or None if anything at all goes wrong."""
        if not self.available:
            return None
        key = hashlib.sha256("\0".join((self.model, system, user)).encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        for attempt in range(4):
            started = time.time()
            try:
                body = self._request("/chat/completions", self._payload(system, user, max_tokens))
                self.calls += 1
                self.latencies.append(time.time() - started)
                self._consecutive_failures = 0
                usage = (body or {}).get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                text = ((body["choices"][0]["message"].get("content")) or "").strip()
                self._cache[key] = text
                return text
            except urllib.error.HTTPError as error:
                detail = _read_error(error)
                self.last_error = f"HTTP {error.code}: {detail[:300]}"
                if error.code == 400 and self._learn_unsupported(detail):
                    continue          # retry immediately with the parameter dropped
                if error.code < 500 and error.code != 429:
                    self._trip()      # bad key/model/request: retrying cannot help
                    return None
                time.sleep(0.5 * (attempt + 1))
            except Exception as error:
                self.last_error = _describe(error)
                time.sleep(0.5 * (attempt + 1))

        self._trip()
        return None

    def _learn_unsupported(self, detail: str) -> bool:
        """Drop a parameter the endpoint rejected so the next try can succeed."""
        lowered = detail.lower()
        for parameter in ("max_tokens", "temperature", "response_format", "seed"):
            if parameter in lowered and parameter not in self._unsupported:
                self._unsupported.add(parameter)
                return True
        return False

    def _trip(self) -> None:
        self.failures += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            # Unreachable or misconfigured (very likely network is disabled).
            # Stop trying; the agent continues deterministically.
            self._disabled = True

    def complete_json(self, system: str, user: str, max_tokens: int = 400) -> dict | None:
        """`complete`, parsed as a JSON object. Tolerates fenced/padded output."""
        text = self.complete(system, user, max_tokens)
        if not text:
            return None
        for candidate in (text, _strip_fence(text), _first_object(text)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def usage_snapshot(self) -> tuple[int, int]:
        return self.prompt_tokens, self.completion_tokens

    def stats(self) -> dict:
        """Everything the submission must disclose about model use."""
        latencies = sorted(self.latencies)
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p95_latency_s": round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else None,
            "dropped_params": sorted(self._unsupported),
            "disabled": self._disabled,
            "last_error": self.last_error,
        }


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _read_error(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", "replace")
    except Exception:
        return str(error)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _first_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else ""


def build_default_client() -> LLMClient | None:
    """The client the Agent uses when none is injected. None when unconfigured."""
    load_dotenv()
    client = LLMClient()
    return client if client.available else None


# --------------------------------------------------------------------------
# Diagnostics: `python3 -m starter.llm_client [--models|--test]`
# --------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    load_dotenv()
    provider, reason = resolve_provider()
    client = LLMClient()
    key = _key_for(client.provider)
    print("provider :", provider or "(none)", f"({reason})")
    print("base_url :", client.base_url)
    print("api_key  :", f"set, {len(key)} chars, ...{key[-4:]}" if key else "NOT SET")
    print("model    :", client.model or "NOT SET")
    blocked = client.why_unavailable()
    print("status   :", blocked or "ready")

    if "--models" in argv:
        if not key:
            print("\ncannot list models without an API key")
            return 1
        print("\nquerying provider for available model ids ...")
        models = client.list_models()
        if models is None:
            print("failed:", client.last_error)
            return 1
        print(f"{len(models)} models available to this key:")
        for name in models:
            print("   ", name)
        return 0

    if "--test" in argv:
        if blocked:
            print("\ncannot test:", blocked)
            return 1
        print("\nsending one extraction call ...")
        started = time.time()
        parsed = client.complete_json(
            "Extract shopping requirements. Reply with JSON only, keys: "
            '{"product_type": string|null, "requirements": [string]}',
            "Customer message:\nHey! I'm after some running shoes, ideally leather and under $80.",
        )
        print(f"elapsed  : {time.time() - started:.2f}s")
        print("parsed   :", json.dumps(parsed) if parsed else f"FAILED ({client.last_error})")
        print("stats    :", json.dumps(client.stats(), indent=2))
        return 0 if parsed else 1

    print("\nrun with --models to list model ids, or --test to make one live call")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
