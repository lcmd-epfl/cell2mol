import signal
from contextlib import contextmanager
import platform


# Define a specific exception for timeouts
class ProcessingTimeoutError(Exception):
    pass


@contextmanager
def set_time_limit(seconds):
    """Context manager to raise a TimeoutError if execution takes too long."""

    def signal_handler(signum, frame):
        raise ProcessingTimeoutError(f"Process timed out after {seconds} seconds")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def set_memory_limit(max_mem_gb):
    """
    Sets a soft limit on memory usage.
    Args:
        max_mem_gb (float): Maximum memory allowed in Gigabytes.
    """
    # The 'resource' module is not available on Windows.
    if platform.system() == "Windows":
        return

    try:
        import resource
    except ImportError:
        return

    # Convert GB to Bytes
    max_mem_bytes = int(max_mem_gb * 1024 * 1024 * 1024)

    # Get current limits (soft, hard)
    # RLIMIT_AS = Address Space limit (Virtual Memory)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    # Set the new limit.
    # The soft limit cannot exceed the hard limit.
    resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, hard))
