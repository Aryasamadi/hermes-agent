from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import httpx

HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
DATA_DIR = HOME / "plugin-data" / "custom-model-manager"
DATA_FILE = DATA_DIR / "models.json"
CONFIG_FILE = HOME / "config.yaml"
ENV_FILE = HOME / ".env"


def _load_registry() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        return {"models": {}}
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"models.json is unreadable; no data was changed: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError("models.json has an invalid structure; no data was changed")
    return data


def _save_registry(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = DATA_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(DATA_FILE)


def _load_config() -> dict:
    # Use Hermes' own config API rather than re-serializing YAML and stripping comments.
    from hermes_cli.config import load_config

    config = load_config()
    if not isinstance(config, dict):
        raise ValueError("Hermes config is not a mapping; no changes were made")
    return config


def _save_config(config: dict) -> None:
    from hermes_cli.config import save_config

    save_config(config)


def _slug(value: str) -> str:
    value = value.strip().lower().replace(" ", "-").replace("/", "-")
    slug = "".join(ch for ch in value if ch.isalnum() or ch in "-_")[:48]
    if not slug:
        raise ValueError("Invalid provider name")
    return slug


def _write_env(name: str, value: str) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    prefix = name + "="
    lines = [line for line in lines if not line.startswith(prefix)]
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")
    lines.append(f'{name}="{escaped}"')
    temp = ENV_FILE.with_suffix(".env.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(ENV_FILE)
    try:
        ENV_FILE.chmod(0o600)
    except OSError:
        pass
    os.environ[name] = value


def _add_provider(name: str, base_url: str, model: str, api_key: str) -> str:
    slug = _slug(name)
    if slug in {"anthropic", "openai", "nvidia", "google", "gemini", "deepseek", "openrouter", "nous", "custom"}:
        raise ValueError(f"'{slug}' is a reserved Hermes provider ID; choose a different provider name")

    env_name = "HERMES_CUSTOM_" + slug.upper().replace("-", "_") + "_API_KEY"
    config = _load_config()
    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("config.yaml providers section is not a mapping; no changes were made")

    # Construct a candidate before writing anything, so validation errors cannot leave half a provider.
    candidate = {
        "name": name,
        "api": base_url.rstrip("/"),
        "key_env": env_name,
        "transport": "openai_chat",
        "default_model": model,
        "discover_models": True,
        "models": {model: {}},
    }
    previous_provider = providers.get(slug)
    previous_record = _load_registry()
    previous_item = previous_record["models"].get(slug)
    previous_env = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else None
    previous_value = os.environ.get(env_name)

    providers[slug] = candidate
    try:
        _save_config(config)
        _write_env(env_name, api_key)
        previous_record["models"][slug] = {
            "name": name,
            "provider": f"custom:{slug}",
            "base_url": base_url.rstrip("/"),
            "model": model,
            "key_env": env_name,
        }
        _save_registry(previous_record)
    except Exception:
        # Roll back all touched state best-effort if any write fails.
        try:
            config = _load_config()
            providers = config.get("providers", {})
            if isinstance(providers, dict):
                if previous_provider is None:
                    providers.pop(slug, None)
                else:
                    providers[slug] = previous_provider
                _save_config(config)
        except Exception:
            pass
        try:
            if previous_env is None:
                ENV_FILE.unlink(missing_ok=True)
            else:
                temp = ENV_FILE.with_suffix(".env.rollback")
                temp.write_text(previous_env, encoding="utf-8")
                temp.replace(ENV_FILE)
                try:
                    ENV_FILE.chmod(0o600)
                except OSError:
                    pass
            if previous_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_value
        except Exception:
            pass
        try:
            registry = _load_registry()
            if previous_item is None:
                registry["models"].pop(slug, None)
            else:
                registry["models"][slug] = previous_item
            _save_registry(registry)
        except Exception:
            pass
        raise
    return slug


def _handle_add(raw_args: str) -> str:
    try:
        args = shlex.split(raw_args)
        if len(args) != 4:
            return 'Usage: /addmodel "Provider Name" "https://example.com/v1" "model-name" "API_KEY"'
        name, base_url, model, api_key = args
        if not base_url.startswith(("http://", "https://")):
            return "❌ API URL must begin with http:// or https://."
        if not model.strip() or not api_key:
            return "❌ Model name and API key cannot be empty."
        slug = _add_provider(name.strip(), base_url, model.strip(), api_key)
        return (
            "✅ Provider saved.\n\n"
            f"Provider: {name}\nModel: {model}\nID: custom:{slug}\n\n"
            f"Test it with: /testmodel {slug}\n"
            f"To make it the global default: /model custom:{slug}:{model} --global"
        )
    except Exception as exc:
        return f"❌ Provider was not added: {exc}"


def _handle_list(raw_args: str = "") -> str:
    try:
        data = _load_registry()
        models = data["models"]
        if not models:
            return "No personal providers are registered. Use /addmodel to add one."
        config = _load_config()
        model_cfg = config.get("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        current_provider = str(model_cfg.get("provider", "auto"))
        current_model = str(model_cfg.get("default", model_cfg.get("model", "not set")))
        lines = [f"Global default: {current_provider} / {current_model}", "", "Registered personal providers:"]
        for slug, item in models.items():
            lines.extend([
                f"• {item.get('name', slug)}",
                f"  ID: {item.get('provider', 'custom:' + slug)}",
                f"  Model: {item.get('model', 'unknown')}",
                f"  Base URL: {item.get('base_url', '')}",
            ])
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Could not read provider registry: {exc}"


async def _handle_test(raw_args: str) -> str:
    try:
        requested = raw_args.strip()
        if not requested:
            return "Usage: /testmodel PROVIDER"
        if requested.startswith("custom:"):
            requested = requested.split(":", 2)[1]
        data = _load_registry()
        item = data["models"].get(_slug(requested))
        if item is None:
            return f"❌ Provider '{requested}' is not registered. Use /mymodel for the exact ID."

        base = item["base_url"].rstrip("/")
        if base.endswith("/chat/completions"):
            endpoint = base
        elif base.endswith("/v1"):
            endpoint = base + "/chat/completions"
        else:
            endpoint = base + "/v1/chat/completions"
        key = os.environ.get(item["key_env"], "")
        if not key:
            return "❌ API key is not loaded. Check the persisted Hermes .env file."

        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": item["model"], "messages": [{"role": "user", "content": "Reply only with OK."}], "max_tokens": 8},
            )
        if response.is_success:
            return f"✅ Model test successful.\nProvider: {item['name']}\nModel: {item['model']}\nHTTP: {response.status_code}"
        # Avoid echoing a provider's full response body into Telegram; it can contain sensitive diagnostics.
        return f"❌ Model test failed. HTTP {response.status_code}. Check the endpoint, key, and model ID."
    except Exception as exc:
        # Do not echo exception details into chat; they may contain endpoint or request diagnostics.
        return f"❌ Model test failed ({type(exc).__name__}). Check the URL, key, model ID, and network."


def _active_default_matches(slug: str, item: dict, config: dict) -> bool:
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        return False
    provider = str(model_cfg.get("provider", "")).strip().lower()
    default_model = str(model_cfg.get("default", model_cfg.get("model", ""))).strip()
    active_url = str(model_cfg.get("base_url", "")).strip().rstrip("/")
    expected_url = str(item.get("base_url", "")).strip().rstrip("/")

    # Exact provider identity is authoritative when Hermes retains it.
    if provider in {slug.lower(), f"custom:{slug}".lower()}:
        return True
    if provider != "custom" or not active_url or active_url != expected_url:
        # In particular, a native provider such as NVIDIA is not the custom provider just
        # because an old base_url field is still present in config.yaml.
        return False

    # The base `custom` provider loses the named alias in some config versions. Only block
    # removal when the saved active model also belongs to this provider, not on URL alone.
    known_models = {str(item.get("model", "")).strip()}
    providers = config.get("providers", {})
    provider_cfg = providers.get(slug, {}) if isinstance(providers, dict) else {}
    if isinstance(provider_cfg, dict):
        models = provider_cfg.get("models", {})
        if isinstance(models, dict):
            known_models.update(str(name).strip() for name in models if name)
        elif isinstance(models, list):
            known_models.update(str(name.get("id", name.get("name", ""))).strip() if isinstance(name, dict) else str(name).strip() for name in models)
    if default_model not in known_models:
        return False

    # If two named providers share this URL and model ID, the active alias cannot be inferred
    # safely. Let the user select a different default instead of deleting a possibly active route.
    if isinstance(providers, dict):
        competitors = 0
        for other_slug, other in providers.items():
            if not isinstance(other, dict):
                continue
            other_url = str(other.get("api", other.get("base_url", ""))).strip().rstrip("/")
            if other_url != active_url:
                continue
            other_models = other.get("models", {})
            model_ids = set(other_models) if isinstance(other_models, dict) else set()
            if default_model in model_ids or default_model == str(other.get("default_model", "")):
                competitors += 1
        if competitors > 1:
            raise ValueError("Multiple registered providers share this URL and model; select another global default first so Hermes can identify the active route safely.")
    return True


def _handle_remove(raw_args: str) -> str:
    try:
        args = shlex.split(raw_args.strip())
        if len(args) != 1:
            return "Usage: /removemodel PROVIDER\nAlias: /removeprovider PROVIDER\nUse /mymodel to see provider IDs."
        requested = args[0].strip()
        if requested.startswith("custom:"):
            parts = requested.split(":")
            if len(parts) >= 2:
                requested = parts[1]
        slug = _slug(requested)
        registry = _load_registry()
        item = registry["models"].get(slug)
        if item is None:
            matches = [(key, value) for key, value in registry["models"].items()
                       if str(value.get("name", "")).strip().casefold() == requested.casefold()]
            if len(matches) != 1:
                return f"❌ Provider '{requested}' is not registered. Use /mymodel. Nothing was changed."
            slug, item = matches[0]

        config = _load_config()
        if _active_default_matches(slug, item, config):
            return (
                "⚠️ این Provider هنوز مدل پیش‌فرض سراسری است؛ برای جلوگیری از قطع ربات حذف نشد.\n\n"
                "ابتدا مدل دیگری را انتخاب کن؛ تغییرها اکنون به‌صورت سراسری ذخیره می‌شوند:\n"
                "/model PROVIDER:MODEL\n\n"
                f"بعد دوباره /removeprovider {slug} را اجرا کن."
            )

        providers = config.get("providers", {})
        if not isinstance(providers, dict):
            raise ValueError("config.yaml providers section is not a mapping; nothing was removed")
        providers.pop(slug, None)
        _save_config(config)

        env_name = str(item.get("key_env", "")).strip()
        if env_name:
            # Do not delete a key still referenced by another registered provider.
            registry_refs = [v for k, v in registry["models"].items() if k != slug and v.get("key_env") == env_name]
            config_refs = [
                v for k, v in providers.items()
                if isinstance(v, dict) and v.get("key_env") == env_name
            ]
            if not registry_refs and not config_refs:
                lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
                kept = [line for line in lines if not line.startswith(env_name + "=")]
                if kept:
                    temp = ENV_FILE.with_suffix(".env.tmp")
                    temp.write_text("\n".join(kept) + "\n", encoding="utf-8")
                    temp.replace(ENV_FILE)
                elif ENV_FILE.exists():
                    ENV_FILE.unlink()
                os.environ.pop(env_name, None)

        del registry["models"][slug]
        _save_registry(registry)
        return f"✅ Provider '{slug}' and its saved model entry were removed. Other history and databases were not touched."
    except Exception as exc:
        return f"❌ Provider was not removed: {exc}"


def register(ctx):
    ctx.register_command("addmodel", _handle_add, description="Add an OpenAI-compatible provider/model", args_hint='<name> <url> <model> <api-key>')
    ctx.register_command("mymodel", _handle_list, description="List personal providers and the current global default")
    ctx.register_command("testmodel", _handle_test, description="Test a registered personal model", args_hint="<provider>")
    ctx.register_command("removemodel", _handle_remove, description="Remove a registered personal provider/model", args_hint="<provider>")
    ctx.register_command("removeprovider", _handle_remove, description="Remove a registered personal provider", args_hint="<provider>")
