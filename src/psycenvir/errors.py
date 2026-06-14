"""Package-specific errors."""


class PsycEnvirError(Exception):
    """Base error for PsycEnvir."""


class TranscriptParseError(PsycEnvirError):
    """Raised when a transcript cannot provide the requested structure."""


class InvalidActionError(PsycEnvirError):
    """Raised when an action cannot be parsed."""


class EnvironmentNotReadyError(PsycEnvirError):
    """Raised when an experiment is specified but lacks recovered dynamics."""
