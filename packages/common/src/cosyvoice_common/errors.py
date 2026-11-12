from __future__ import annotations

from typing import Any


class APIError(Exception):
    """An error that can be rendered as a Bailian-compatible error."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        *,
        retryable: bool = False,
        retry_after: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.details = details

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if request_id:
            body["request_id"] = request_id
        if self.details is not None:
            body["details"] = self.details
        return body


def invalid_parameter(message: str, **kwargs: Any) -> APIError:
    return APIError("InvalidParameter", message, 400, **kwargs)


def voice_not_found(message: str) -> APIError:
    return APIError("VoiceNotFound", message, 400)


def unsupported_format(message: str) -> APIError:
    return APIError("UnsupportedFormat", message, 400)


def unsupported_parameter(message: str) -> APIError:
    return APIError("UnsupportedParameter", message, 400)


def text_too_long(message: str) -> APIError:
    return APIError("TextTooLong", message, 400)


def unauthorized(message: str = "invalid API key") -> APIError:
    return APIError("Unauthorized", message, 401)


def throttled(message: str = "too many concurrent streams") -> APIError:
    return APIError("Throttling", message, 429, retryable=True, retry_after=1)


def timeout_error(message: str = "upstream timed out") -> APIError:
    return APIError("Timeout", message, 504, retryable=True)


def service_unavailable(message: str = "model is not ready") -> APIError:
    return APIError("ServiceUnavailable", message, 503, retryable=True)


def internal_error(message: str = "internal error") -> APIError:
    return APIError("InternalError", message, 500)


def normalize_error(exc: Exception) -> APIError:
    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, TimeoutError):
        return timeout_error()
    return internal_error(str(exc) or exc.__class__.__name__)
