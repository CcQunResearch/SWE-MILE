"""Bound cancellation handshakes without hiding their eventual exception."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


async def run_blocking_to_completion(function, *args, **kwargs):
    """Keep the control loop responsive without abandoning a stateful RPC."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rllm-backend")
    future = executor.submit(copy_context().run, function, *args, **kwargs)
    cancellation = None
    try:
        while not future.done():
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError as exc:
                cancellation = exc
        if cancellation is not None:
            try:
                future.result()
            except BaseException as exc:
                raise cancellation from exc
            raise cancellation
        return future.result()
    finally:
        executor.shutdown(wait=False)


def run_with_bounded_cleanup(coroutine, timeout=120.0):
    """Return to the owning Ray/backend shutdown even if cancellation stalls.

    The backend owns process reclamation. asyncio.run would otherwise wait
    again for detached tasks, undoing the earlier bounded cleanup decision.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    failed = False
    try:
        return loop.run_until_complete(coroutine)
    except BaseException:
        failed = True
        raise
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        pending.add(loop.create_task(loop.shutdown_asyncgens()))
        try:
            done, remaining = loop.run_until_complete(asyncio.wait(pending, timeout=timeout))
            for task in done:
                if not task.cancelled():
                    task.exception()
            if remaining:
                logging.getLogger(__name__).error("Training loop cleanup timed out; backend must reclaim %d pending tasks", len(remaining))
                if not failed:
                    raise TimeoutError("Training event loop cleanup deadline exceeded")
        finally:
            loop.close()
            asyncio.set_event_loop(None)


async def bounded_cleanup(awaitable, timeout=120.0):
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if done:
        return task.result()
    task.cancel()
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    raise TimeoutError(f"Training cleanup exceeded {timeout:g}s")
