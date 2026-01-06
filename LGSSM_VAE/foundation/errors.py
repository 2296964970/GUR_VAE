from __future__ import annotations


class LGSSMVAEError(Exception):
    """Base exception for this project.

    Library code should raise these exceptions; CLI code should translate them
    into a user-facing exit message (SystemExit) when appropriate.
    """


class ValidationError(LGSSMVAEError):
    """Raised when structured input (config/ckpt/data) fails validation."""


class ConfigError(ValidationError):
    """Raised when config.yaml (after extends merge) is invalid."""


class CheckpointError(ValidationError):
    """Raised when checkpoint metadata/state is invalid or incomplete."""


class DataError(ValidationError):
    """Raised when input data files/arrays violate expected constraints."""


class InferenceError(ValidationError):
    """Raised when inference inputs/requests are invalid."""


__all__ = [
    "LGSSMVAEError",
    "ValidationError",
    "ConfigError",
    "CheckpointError",
    "DataError",
    "InferenceError",
]
