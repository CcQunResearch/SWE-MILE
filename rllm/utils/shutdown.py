"""Keep training failures primary while attempting all independent cleanup."""

import logging

from rllm.utils.infrastructure_diagnostics import infrastructure_exception_chain

logger = logging.getLogger(__name__)


class ShutdownError(RuntimeError):
    def __init__(self, errors):
        self.diagnostics = {"shutdown_errors": errors}
        super().__init__("Trainer shutdown failed: " + ", ".join(error["component"] for error in errors))


def shutdown_components(components):
    errors = []
    for name, callback in components:
        try:
            callback()
        except BaseException as exc:
            errors.append({"component": name, "exception_chain": infrastructure_exception_chain(exc)})
            # Nested cleanup wrappers already carry every causal exception.
            # Retain that structured chain without printing it at each layer.
            from rllm.utils.diagnostic_events import emit_diagnostic
            emit_diagnostic("shutdown_failure", component=name, exception_chain=errors[-1]["exception_chain"])
            logger.error("Shutdown failed for %s; continuing remaining cleanup: %s: %s", name, type(exc).__name__, exc)
    if errors:
        raise ShutdownError(errors)


def shutdown_preserving_error(trainer, primary_error):
    try:
        trainer.shutdown()
    except BaseException as cleanup_error:
        detail = infrastructure_exception_chain(cleanup_error)
        if primary_error is not None:
            primary_error.shutdown_errors = detail
            primary_error.add_note("Additional shutdown failure: " + str(cleanup_error))
        record = getattr(trainer, "record_shutdown_failure", None)
        if callable(record):
            try:
                record(primary_error, cleanup_error)
            except Exception:
                logger.exception("Could not persist shutdown failure")
        if primary_error is None:
            raise
        logger.error("Preserving training exception after shutdown failure: %s", cleanup_error)
