"""A package providing retry utility functionality."""

from collections.abc import Callable

from tenacity import retry, stop_after_attempt, wait_random_exponential


def random_exponential_retry[T](
    func: Callable[[], T], max_wait: int = 10, max_attempts: int = 3
) -> T:
    """Execute the given function with a random exponential retry strategy.

    Args:
        func: The function to be executed.
        max_wait: The maximum wait time in seconds.
        max_attempts: The maximum retry attempts.

    Returns: The return value of the specified function.

    Raises:
        Exception: If maximum retry attempts failed.

    """

    @retry(
        wait=wait_random_exponential(max=max_wait),
        stop=stop_after_attempt(max_attempts),
    )
    def execute() -> T:
        return func()

    return execute()
