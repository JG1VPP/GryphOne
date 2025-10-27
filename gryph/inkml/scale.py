from functools import partial
from typing import Tuple

from more_itertools import collapse, transpose


def scale_point(point, origin: Tuple[float, float], scale: float):
    return tuple(scale * (p - m) for p, m in zip(point, origin))


def scale_trace(trace, origin: Tuple[float, float], scale: float):
    return tuple(scale_point(xy, origin, scale) for xy in trace)


def scale_inkml(traces, w: int, h: int):
    x, y = transpose(collapse(traces, levels=1))

    # determine shape
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    # determine scale
    x_scale = w / (x_max - x_min)
    y_scale = h / (y_max - y_min)

    # fit formula on screen
    scale = min(x_scale, y_scale)

    # preserve aspect ratio
    f = partial(scale_trace, scale=scale)
    f = partial(f, origin=(x_min, y_min))

    return tuple(map(f, traces))
