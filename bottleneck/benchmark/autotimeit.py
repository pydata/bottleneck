import timeit
from typing import Tuple


def autotimeit(
    stmt: str, setup: str = "pass", repeat: int = 3, mintime: float = 0.2
) -> float:
    timer = timeit.Timer(stmt, setup)
    number, time1 = autoscaler(timer, mintime)
    time2 = timer.repeat(repeat=repeat - 1, number=number)
    return min(time2 + [time1]) / number


def autoscaler(timer: timeit.Timer, mintime: float) -> Tuple[int, float]:
    number = 1
    for _ in range(12):
        time = timer.timeit(number)
        if time > mintime:
            return number, time
        number *= 10
    raise RuntimeError("function is too fast to test")
