import os
from dataclasses import dataclass

from tars.core import CLAUDE_MODELS, DEFAULT_MODEL, parse_model

_ALLOWED_ROUTING_POLICIES = {"tool"}


@dataclass(frozen=True)
class ModelConfig:
    primary_provider: str
    primary_model: str
    remote_provider: str | None
    remote_model: str | None
    routing_policy: str = "tool"


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "":
        return None
    if cleaned.lower() in {"none", "null"}:
        return None
    return cleaned


def _coalesce(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _validate_routing_policy(policy: str) -> None:
    if policy not in _ALLOWED_ROUTING_POLICIES:
        raise ValueError(f"Unknown routing policy: {policy}")


def _validate_remote_model(provider: str, model: str) -> None:
    if provider != "claude":
        return
    if model in CLAUDE_MODELS:
        raise ValueError(
            f"Remote Claude model must be explicit (got {model!r}). "
            "Use a versioned model id like 'claude-sonnet-4-5-20250929'."
        )


def load_model_config() -> ModelConfig:
    default_model_env = _clean_env(os.environ.get("TARS_DEFAULT_MODEL"))
    legacy_model_env = _clean_env(os.environ.get("TARS_MODEL"))
    model_str = _coalesce(default_model_env, legacy_model_env)
    if model_str is None:
        model_str = DEFAULT_MODEL
    primary_provider, primary_model = parse_model(model_str)

    remote_env_raw = os.environ.get("TARS_REMOTE_MODEL")
    remote_env = _clean_env(remote_env_raw)
    legacy_remote_env = _clean_env(os.environ.get("TARS_ESCALATION_MODEL"))
    if remote_env_raw is not None and remote_env is None:
        remote_str = None
    else:
        remote_str = _coalesce(remote_env, legacy_remote_env)
    remote_provider = None
    remote_model = None
    if remote_str is not None:
        remote_provider, remote_model = parse_model(remote_str)
        _validate_remote_model(remote_provider, remote_model)

    policy_env = _clean_env(os.environ.get("TARS_ROUTING_POLICY"))
    routing_policy = policy_env if policy_env is not None else "tool"
    _validate_routing_policy(routing_policy)

    return ModelConfig(
        primary_provider=primary_provider,
        primary_model=primary_model,
        remote_provider=remote_provider,
        remote_model=remote_model,
        routing_policy=routing_policy,
    )


def apply_cli_overrides(config: ModelConfig, model: str | None, remote_model: str | None) -> ModelConfig:
    primary_provider = config.primary_provider
    primary_model = config.primary_model
    if model is not None and model != "":
        primary_provider, primary_model = parse_model(model)

    remote_provider = config.remote_provider
    remote_model_value = config.remote_model
    if remote_model is not None and remote_model != "":
        remote_provider, remote_model_value = parse_model(remote_model)
        _validate_remote_model(remote_provider, remote_model_value)

    return ModelConfig(
        primary_provider=primary_provider,
        primary_model=primary_model,
        remote_provider=remote_provider,
        remote_model=remote_model_value,
        routing_policy=config.routing_policy,
    )


def model_summary(config: ModelConfig) -> dict[str, str]:
    remote = "none"
    if config.remote_provider is not None and config.remote_model is not None:
        remote = f"{config.remote_provider}:{config.remote_model}"
    return {
        "primary": f"{config.primary_provider}:{config.primary_model}",
        "remote": remote,
        "routing_policy": config.routing_policy,
    }
