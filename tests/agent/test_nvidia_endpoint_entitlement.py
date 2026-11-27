"""Regression coverage for NVIDIA NIM account-scoped Function 404 responses."""

from agent.error_classifier import FailoverReason, classify_api_error
from agent.error_surface import build_error_surface_from_result
from agent.turn_failure_copy import nonretryable_copy


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "mistralai/mistral-large-4-1t-a52b"


class _APIError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code
        self.body = {}


def _function_not_found_error() -> _APIError:
    return _APIError(
        "HTTP 404: Function 'f238fb4a-8a45-473f-a81f-b7d30f56e2b0': "
        "Not found for account 'account-hash'",
        status_code=404,
    )

def test_nvidia_account_function_404_is_terminal_and_falls_back_without_retry():
    error = _function_not_found_error()
    classified = classify_api_error(
        error, provider="custom", model=NVIDIA_MODEL, base_url=NVIDIA_BASE_URL,
    )

    assert classified.reason == FailoverReason.provider_policy_blocked
    assert classified.retryable is False
    assert classified.should_fallback is True
    assert classified.should_rotate_credential is False
    assert classified.error_context == {"provider_access": "nvidia_public_api_endpoints"}

    surface = build_error_surface_from_result(
        {
            "completed": False,
            "failed": True,
            "error": str(error),
            "failure_reason": classified.reason.value,
            "failure_retryable": classified.retryable,
        },
        provider="custom",
        model=NVIDIA_MODEL,
    )
    assert surface["code"] == "provider_policy_blocked"
    assert surface["retryable"] is False


def test_nvidia_account_function_404_copy_explains_account_access_not_bad_payload():
    error = _function_not_found_error()
    classified = classify_api_error(
        error, provider="custom", model=NVIDIA_MODEL, base_url=NVIDIA_BASE_URL,
    )

    message = nonretryable_copy(
        classified, provider="custom", model=NVIDIA_MODEL, summary=str(error),
    )
    assert "Public API Endpoints" in message
    assert "account/Function mapping" in message
    assert "Repeating the same request will not fix it" in message
    assert "malformed" not in message.lower()


def test_same_function_404_on_unrelated_custom_host_is_not_misclassified():
    error = _function_not_found_error()
    classified = classify_api_error(
        error, provider="custom", model="my-model", base_url="https://proxy.example/v1",
    )

    assert classified.reason == FailoverReason.unknown
    assert classified.retryable is True
    assert classified.should_fallback is False
