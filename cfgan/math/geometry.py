from typing import NamedTuple, Optional


class Point(NamedTuple):
    x: float
    y: float


class Line(NamedTuple):
    coefficient: float
    point: Point


def get_intersection(l1: Line, l2: Line) -> Optional[Point]:
    if l1.coefficient == l2.coefficient:
        return None
    else:
        x = (l2.point.y - l1.point.y + l1.coefficient * l1.point.x - l2.coefficient * l2.point.x) / (
            l1.coefficient - l2.coefficient
        )
        y = l1.coefficient * (x - l1.point.x) + l1.point.y
        return Point(x, y)
