"""Service integrations for the chat application."""


class ToolExecutionError(Exception):
    """A safe error to return from a model-callable tool."""

    def __init__(self, code: str, message: str, *, options: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.options = options or []
