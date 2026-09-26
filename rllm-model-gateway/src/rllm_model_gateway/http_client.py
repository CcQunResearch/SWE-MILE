"""Share immutable TLS configuration, never connections, across rollout clients."""

import ssl
import threading

import httpx

_context: ssl.SSLContext | None = None
_context_lock = threading.Lock()


def shared_ssl_context() -> ssl.SSLContext:
    """Load the environment's CA configuration once per process.

    Even HTTP-only httpx clients initialize TLS contexts. Hundreds of rollout
    threads otherwise load the same CA bundle concurrently at every model
    switch. Keep verification enabled and retain httpx's SSL_CERT_* handling.
    """
    global _context
    with _context_lock:
        if _context is None:
            _context = httpx.create_ssl_context()
        return _context
