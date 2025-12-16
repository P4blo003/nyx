
# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio


# ==============================
# FUNCTIONS
# ==============================

async def cancel_task(task_to_cancel:asyncio.Task|None) -> None:
    """
    Cancel the given task.
    """
    # Checks if it's None.
    if task_to_cancel is None: return

    # Checks if the task is done.
    if task_to_cancel.done(): return

    # Signal cancellation.
    task_to_cancel.cancel()

    # Try-Except to manage errors.
    try:
        # Awaits for the task.
        await task_to_cancel

    # If task is cancelled.
    except asyncio.CancelledError: pass
    # Executes finally.
    finally: return
