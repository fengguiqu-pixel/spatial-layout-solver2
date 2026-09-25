"""几何基础库：向量、多边形、有向包围盒（OBB）以及各类相交判定。

约定
----
* 坐标单位与输入数据一致（题目数据为 mm）。
* 所有"接触 / 重叠"判定都带容差，避免浮点误差把"贴墙摆放"误判成重叠。
* ``point_in_polygon`` 用奇偶规则，对题目里出现的自接触 / 重复边（如 example4
  左墙被重复描了两遍）也能给出稳定的内外判定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
EPS = 1e-9


# --------------------------------------------------------------------------
# 向量
# --------------------------------------------------------------------------
def vsub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def vadd(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def vmul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def vdot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def vcross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def vlen(a: Point) -> float:
    return math.hypot(a[0], a[1])


def vdist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def vunit(a: Point) -> Point:
    l = vlen(a)
    return (a[0] / l, a[1] / l) if l > EPS else (0.0, 0.0)


def rotate_point(p: Point, deg: float, origin: Point = (0.0, 0.0)) -> Point:
    """将点 p 绕 origin 逆时针旋转 deg 度。"""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y = p[0] - origin[0], p[1] - origin[1]
    return (origin[0] + x * c - y * s, origin[1] + x * s + y * c)


def edge_angle_deg(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def norm_angle_180(deg: float) -> float:
    """归一化到 (-180, 180]。"""
    d = deg % 360.0
    if d > 180.0:
        d -= 360.0
    return d


# --------------------------------------------------------------------------
# 有向包围盒 OBB
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OBB:
    """矩形物体：中心 + 局部半长 + 旋转角（度，局部 x 轴相对世界 x 轴，逆时针）。"""

    cx: float
    cy: float
    hw: float
    hh: float
    angle: float = 0.0

    @property
    def center(self) -> Point:
        return (self.cx, self.cy)

    @property
    def size(self) -> Tuple[float, float]:
        return (2.0 * self.hw, 2.0 * self.hh)

    def corners(self) -> List[Point]:
        base = [(-self.hw, -self.hh), (self.hw, -self.hh), (self.hw, self.hh), (-self.hw, self.hh)]
        return [rotate_point((self.cx + dx, self.cy + dy), self.angle, (self.cx, self.cy))
                for dx, dy in base]

    def edges(self) -> List[Tuple[Point, Point]]:
        c = self.corners()
        return [(c[i], c[(i + 1) % 4]) for i in range(4)]

    def corner(self, sx: int, sy: int) -> Point:
        """取局部坐标角点，sx / sy ∈ {-1, 1}。"""
        return rotate_point((self.cx + sx * self.hw, self.cy + sy * self.hh),
                            self.angle, (self.cx, self.cy))

    def area(self) -> float:
        return 4.0 * self.hw * self.hh

    def rotated(self, deg: float) -> "OBB":
        """整体绕原点旋转 deg 度（矩形旋转后仍是矩形）。"""
        c = rotate_point((self.cx, self.cy), deg)
        return OBB(c[0], c[1], self.hw, self.hh, self.angle + deg)


def obb_from_aabb(rect: "Rect") -> OBB:
    """由轴对齐矩形（见 Rect）构造 OBB。"""
    return OBB((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0,
               (rect.x1 - rect.x0) / 2.0, (rect.y1 - rect.y0) / 2.0, 0.0)


@dataclass(frozen=True)
class Rect:
    """轴对齐矩形（在某一旋转坐标系下使用），x0 <= x1, y0 <= y1。"""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Point:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def as_obb(self) -> OBB:
        return obb_from_aabb(self)


# --------------------------------------------------------------------------
# 多边形
# --------------------------------------------------------------------------
def polygon_edges(poly: Sequence[Point]) -> List[Tuple[Point, Point]]:
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


def signed_area(poly: Sequence[Point]) -> float:
    return 0.5 * sum(vcross(a, b) for a, b in polygon_edges(poly))


def polygon_area(poly: Sequence[Point]) -> float:
    """轮廓面积（取绝对值，与顶点绕向无关）。"""
    return abs(signed_area(poly))


def polygon_bbox(poly: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def clean_polygon(pts: Sequence[Point], tol: float = 1e-6) -> List[Point]:
    """去掉重复点、闭合点；不改动顶点顺序（顺序对奇偶规则不重要）。"""
    out: List[Point] = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        if out and vdist(out[-1], p) <= tol:
            continue
        out.append(p)
    if len(out) > 1 and vdist(out[0], out[-1]) <= tol:
        out.pop()
    return out


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 <= EPS:
        return vdist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def point_in_polygon(p: Point, poly: Sequence[Point], tol: float = 1e-6) -> int:
    """1 = 内部，0 = 边界（容差内），-1 = 外部。"""
    for a, b in polygon_edges(poly):
        if point_segment_distance(p, a, b) <= tol:
            return 0
    x, y = p
    inside = False
    for a, b in polygon_edges(poly):
        if (a[1] > y) != (b[1] > y):
            t = (y - a[1]) / (b[1] - a[1])
            if a[0] + t * (b[0] - a[0]) > x:
                inside = not inside
    return 1 if inside else -1


def point_inside_or_on(p: Point, poly: Sequence[Point], tol: float = 1e-6) -> bool:
    return point_in_polygon(p, poly, tol) >= 0


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


# 矩形边相对旋转后多边形边的"贴边"容差（mm）。
# 贴墙摆放时 gen_positions 把矩形边精确压在 wall.coord 上，但旋转后的多边形实际边
# 有亚毫米级微斜，矩形竖边端点会高出多边形边约 1e-4 mm，被"穿边检查"误判为越界。
# 允许矩形边在 EDGE_GRAZE_TOL 范围内贴着墙边，既消除该误判，又不让物品真正探出房间。
EDGE_GRAZE_TOL = 0.1


def segments_properly_cross(a: Point, b: Point, c: Point, d: Point, tol: float = 1e-9) -> bool:
    """两线段是否"穿透式"相交（交点在两者的内部）。

    共线重叠、端点接触都返回 False —— 贴墙摆放时矩形边与墙共线，必须放行。
    tol 为允许的两线段"擦边"距离（按线段长度归一化的带符号距离，单位与坐标一致）：
    若某端点落在对向线段所在直线 tol 范围内，视为接触而非穿透。
    """
    L2_cd = (d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2
    if L2_cd < 1e-18:
        return False
    k_cd = 1.0 / (2.0 * L2_cd ** 0.5)
    sa = _orient(c, d, a) * k_cd
    sb = _orient(c, d, b) * k_cd
    if abs(sa) <= tol and abs(sb) <= tol:
        return False
    if not ((sa > tol and sb < -tol) or (sa < -tol and sb > tol)):
        return False
    L2_ab = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    if L2_ab < 1e-18:
        return False
    k_ab = 1.0 / (2.0 * L2_ab ** 0.5)
    sc = _orient(a, b, c) * k_ab
    sd = _orient(a, b, d) * k_ab
    if abs(sc) <= tol and abs(sd) <= tol:
        return False
    if not ((sc > tol and sd < -tol) or (sc < -tol and sd > tol)):
        return False
    return True


def rect_inside_polygon(obb: OBB, poly: Sequence[Point], tol: float = 1e-6) -> bool:
    """矩形是否完全落在多边形内（允许贴边，容差 tol）。"""
    for c in obb.corners():
        if point_in_polygon(c, poly, tol) < 0:
            return False
    for e in obb.edges():
        for pe in polygon_edges(poly):
            if segments_properly_cross(e[0], e[1], pe[0], pe[1], EDGE_GRAZE_TOL):
                return False
    return True


# --------------------------------------------------------------------------
# 矩形 / 矩形
# --------------------------------------------------------------------------
def _proj_range(corners: Sequence[Point], ax: Point) -> Tuple[float, float]:
    vals = [vdot(c, ax) for c in corners]
    return min(vals), max(vals)


def obb_penetration(a: OBB, b: OBB) -> float:
    """分离轴定理：返回最小穿透深度。<= 0 表示分离或仅接触。"""
    axes: List[Point] = []
    for r in (a, b):
        rad = math.radians(r.angle)
        axes.append((math.cos(rad), math.sin(rad)))
        axes.append((-math.sin(rad), math.cos(rad)))
    ca, cb = a.corners(), b.corners()
    best = float("inf")
    for ax in axes:
        a0, a1 = _proj_range(ca, ax)
        b0, b1 = _proj_range(cb, ax)
        best = min(best, min(a1, b1) - max(a0, b0))
        if best <= 0.0:
            return best
    return best


def obb_overlap(a: OBB, b: OBB, tol: float = 1.0) -> bool:
    """是否重叠（穿透深度 > tol 才算重叠；仅"贴着"不算）。"""
    return obb_penetration(a, b) > tol


def rect_rect_contact(a: Rect, b: Rect, tol: float = 1.0) -> float:
    """两个轴对齐矩形的接触长度（贴着但没压进去）。没接触返回 0。"""
    ox = min(a.x1, b.x1) - max(a.x0, b.x0)
    oy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ox < -tol or oy < -tol:
        return 0.0
    if abs(ox) <= tol and oy > tol:
        return oy
    if abs(oy) <= tol and ox > tol:
        return ox
    return 0.0


def obb_edge_contact_length(obb: OBB, a: Point, b: Point, tol: float = 1.0) -> float:
    """矩形与线段（墙）的接触长度；不接触返回 0。"""
    d = vsub(b, a)
    L = vlen(d)
    if L <= EPS:
        return 0.0
    u = (d[0] / L, d[1] / L)
    n = (-u[1], u[0])
    ns = [vdot(c, n) for c in obb.corners()]
    wn = vdot(a, n)
    if min(abs(min(ns) - wn), abs(max(ns) - wn)) > tol:
        return 0.0
    us = [vdot(c, u) for c in obb.corners()]
    ea, eb = sorted((vdot(a, u), vdot(b, u)))
    return max(0.0, min(max(us), eb) - max(min(us), ea))


# --------------------------------------------------------------------------
# 线段 / 矩形（用于"是否挡门"）
# --------------------------------------------------------------------------
def segment_intersects_rect(p: Point, q: Point, obb: OBB, tol: float = 0.0) -> bool:
    """线段（含端点、含共线重叠）是否与矩形闭合区域相交。Liang–Barsky 裁剪。"""
    c = (obb.cx, obb.cy)
    lp = rotate_point(p, -obb.angle, c)
    lq = rotate_point(q, -obb.angle, c)
    x0, y0 = lp[0] - c[0], lp[1] - c[1]
    dx, dy = lq[0] - lp[0], lq[1] - lp[1]
    hw, hh = obb.hw + tol, obb.hh + tol
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 + hw), (dx, hw - x0), (-dy, y0 + hh), (dy, hh - y0)):
        if abs(pp) < 1e-12:
            if qq < 0.0:
                return False
            continue
        r = qq / pp
        if pp < 0.0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return t0 <= t1 + 1e-12


def ray_distance_to_boundary(poly: Sequence[Point], p: Point, d: Point, maxd: float = 1e9) -> float:
    """从 p 沿单位方向 d 走到多边形边界的距离（走不到就返回 maxd）。"""
    best = maxd
    for a, b in polygon_edges(poly):
        # 解 p + t*d = a + s*(b-a)
        e = vsub(b, a)
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        rhs = vsub(a, p)
        t = (rhs[0] * e[1] - rhs[1] * e[0]) / den
        s = (rhs[0] * d[1] - rhs[1] * d[0]) / den
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
            best = min(best, t)
    return best
