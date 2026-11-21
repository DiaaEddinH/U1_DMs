from math import pi
from numpy.typing import NDArray

def wrap(array: NDArray | float) -> NDArray | float:
    return (array + pi) % (2 * pi) - pi