from supernova._logger import get_logger

logger = get_logger(__name__)

try:
    import numba
except ImportError:
    logger.warning("Numba is not installed. Some functions may be slower. Install numba for better performance.")
    def njit(*args, **kwargs):
        """Dummy njit decorator that does nothing."""
        def decorator(func):
            return func
        return decorator
    
    prange = range
else:
    from numba import njit, prange

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False