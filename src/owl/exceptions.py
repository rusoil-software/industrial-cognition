import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Call this inside create_application() to attach global handlers.
    Domain-specific HTTPExceptions bubble up naturally via FastAPI,
    so we only need to catch the things that fall through.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
            request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Override FastAPI's default 422 handler to log the errors
        and return a cleaner, consistent error envelope.
        """
        logger.warning("Validation error on %s: %s", request.url, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "VALIDATION_ERROR",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
            request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for anything that slips past domain handlers.
        Logs the full traceback but returns a safe, opaque message to the client.
        """
        logger.exception("Unhandled exception on %s", request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "INTERNAL_SERVER_ERROR", "detail": "An unexpected error occurred."},
        )