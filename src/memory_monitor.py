import os
import psutil
from functools import wraps
from time import time
from typing import Callable, Any


def get_process_memory() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def log_memory_usage(func: Callable) -> Callable:
    """Decorator to log memory usage before and after function execution."""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_mem = get_process_memory()
        start_time = time()

        print(f"\nStarting {func.__name__}")
        print(f"Initial memory usage: {start_mem:.2f} MB")

        try:
            result = await func(*args, **kwargs)

            end_mem = get_process_memory()
            end_time = time()

            print(f"\nFinished {func.__name__}")
            print(f"Final memory usage: {end_mem:.2f} MB")
            print(f"Memory change: {end_mem - start_mem:.2f} MB")
            print(f"Time taken: {end_time - start_time:.2f} seconds")

            return result

        except Exception as e:
            print(f"\nError in {func.__name__}: {str(e)}")
            print(f"Memory at error: {get_process_memory():.2f} MB")
            raise

    return wrapper


# Usage example:
# @log_memory_usage
# async def generate_story(story_name: str):
#     ...
