from dataclasses import dataclass
from typing import Iterable, List, Optional

@dataclass
class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class Segment:
    pass

class PathEl:
    def to_svg_chunk(self) -> str:
        raise NotImplementedError
    pass

class MoveTo(PathEl):
    def __init__(self, pt: Point):
        self.pt = pt
    
    def to_svg_chunk(self) -> str:
        return f"M {self.pt.x} {self.pt.y}"

class LineTo(PathEl):
    def __init__(self, pt: Point):
        self.pt =pt
    
    def to_svg_chunk(self) -> str:
        return f"L {self.pt.x} {self.pt.y}"

class CurveTo(PathEl):
    def __init__(self, pt1: Point, pt2: Point, pt3: Point):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
    
    def to_svg_chunk(self) -> str:
        return f"C {self.pt1.x} {self.pt1.y}, {self.pt2.x} {self.pt2.y}, {self.pt3.x} {self.pt3.y}"

class Close(PathEl):
    def to_svg_chunk(self) -> str:
        return "Z"


class BezPath:
    def __init__(self, *args, **kwargs):
        self._elements: List[PathEl] = []
    
    def push(self, el: PathEl):
        self._elements.append(el)
    
    def elements(self) -> Iterable[PathEl]:
        return iter(self._elements)
    
    def move_to(self, pt: Point):
        self.push(MoveTo(pt))

    def curve_to(self, pt1: Point, pt2: Point, pt3: Point):
        self.push(CurveTo(pt1, pt2, pt3))
    
    def line_to(self, pt: Point):
        self.push(LineTo(pt))

    def close_path(self):
        self.push(Close())

    def to_svg(self) -> str:
        return " ".join(el.to_svg_chunk() for el in self._elements)
    
    def segments(self) -> Iterable[Segment]:
        start_pt: Optional[Point] = None
        for el in self._elements:
            if isinstance(el, MoveTo):
                start_pt = el.pt
            elif isinstance(el, LineTo):
                if not start_pt:
                    raise ValueError("LineTo without a starting point")
                yield Line(start_pt, el.pt)
                start_pt = el.pt
            elif isinstance(el, CurveTo):
                if not start_pt:
                    raise ValueError("CurveTo without a starting point")
                yield CubicBez(start_pt, el.pt1, el.pt2, el.pt3)
                start_pt = el.pt3
            elif isinstance(el, Close):
                pass

@dataclass
class CubicBez(Segment):
    p0: Point   
    p1: Point
    p2: Point
    p3: Point

    def eval(self, t: float) -> Point:
        """Evaluate the cubic Bezier curve at parameter t (0 <= t <= 1)."""
        u = 1 - t
        x = (u**3 * self.p0.x +
             3 * u**2 * t * self.p1.x +
             3 * u * t**2 * self.p2.x +
             t**3 * self.p3.x)
        y = (u**3 * self.p0.y +
             3 * u**2 * t * self.p1.y +
             3 * u * t**2 * self.p2.y +
             t**3 * self.p3.y)
        return Point(x, y)


    def start(self) -> Point:
        return self.p0
    
    def end(self) -> Point:
        return self.p1
    
@dataclass
class Line(Segment):
    p0: Point
    p1: Point

    def start(self) -> Point:
        return self.p0
    
    def end(self) -> Point:
        return self.p1
