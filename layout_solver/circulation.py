"""动线（circulation）：从入口门到出口门（或房间最深处）的人行通道。

做法
----
1. 把房间打成栅格，标出可通行的格子（在轮廓内、不在内开门禁放区里）；
2. 用带"离墙惩罚"的 Dijkstra 求一条**尽量走中间**的路径
   （贴着墙走的成本更高，这样算出来的通道自然落在房间中轴）；
3. 只有一个门时，终点取"从门出发走最远的可达点"（房间最深处）；
4. 路径做折线简化后，膨胀成指定宽度的带状区域，作为**硬禁放区**。

这样物品的摆放结果天然是「贴墙 + 分列通道两侧」，中间留出人走的通道。
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

from . import config as cfg
from .geometry import (OBB, Point, point_in_polygon, point_segment_distance, polygon_bbox,
                       polygon_edges, rotate_point, vadd, vdist, vmul, vsub, vunit)

INF = float("inf")


# ---------------------------------------------------------------------------
# 栅格
# ---------------------------------------------------------------------------
def build_grid(scene, target_cells: int = 12000):
    """返回 (free, wall_dist, n, m, x0, y0, cw, ch)。

    free[i][j] 为 True 表示可通行；wall_dist[i][j] 是该格到最近墙的距离。
    """
    x0, y0, x1, y1 = polygon_bbox(scene.polygon)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    n = max(10, int(round(math.sqrt(target_cells * w / h))))
    m = max(10, int(round(target_cells / n)))
    cw, ch = w / n, h / m

    blocks = [r.corners() for r in scene.reserved]
    edges = polygon_edges(scene.polygon)
    free = [[False] * m for _ in range(n)]
    wall = [[0.0] * m for _ in range(n)]

    for i in range(n):
        x = x0 + (i + 0.5) * cw
        for j in range(m):
            y = y0 + (j + 0.5) * ch
            if point_in_polygon((x, y), scene.polygon, 1e-6) != 1:
                continue
            blocked = False
            for c in blocks:
                if point_in_polygon((x, y), c, 1e-6) >= 0:
                    blocked = True
                    break
            if blocked:
                continue
            free[i][j] = True
            wall[i][j] = min(point_segment_distance((x, y), a, b) for a, b in edges)
    return free, wall, n, m, x0, y0, cw, ch


def _nearest_free(free, n, m, x0, y0, cw, ch, p: Point) -> Optional[Tuple[int, int]]:
    i0 = min(n - 1, max(0, int((p[0] - x0) / cw)))
    j0 = min(m - 1, max(0, int((p[1] - y0) / ch)))
    if free[i0][j0]:
        return (i0, j0)
    best = None
    best_d = INF
    for r in range(1, max(n, m)):
        for di in range(-r, r + 1):
            for dj in (-r, r):
                for cand in ((i0 + di, j0 + dj), (i0 + dj, j0 + di)):
                    i, j = cand
                    if 0 <= i < n and 0 <= j < m and free[i][j]:
                        d = (i - i0) ** 2 + (j - j0) ** 2
                        if d < best_d:
                            best_d = d
                            best = (i, j)
        if best:
            return best
    return None


# ---------------------------------------------------------------------------
# 最短路
# ---------------------------------------------------------------------------
def _wall_penalty(dw: float, mode: str) -> float:
    """离墙越近成本越高；center 模式惩罚更强，逼着通道往中间走。"""
    if mode == "shortest":
        return 1.0
    if mode == "side":       # 允许贴着一侧走（离墙 400 左右最舒服）
        return 1.0 + cfg.AISLE_WALL_PENALTY * min(1.0, abs(dw - cfg.AISLE_MIN_WALL_DIST)
                                                  / max(cfg.AISLE_MIN_WALL_DIST, 1.0))
    return 1.0 + cfg.AISLE_WALL_PENALTY * max(
        0.0, 1.0 - dw / max(cfg.AISLE_CENTER_DIST, 1.0))


def _dijkstra(free, wall, n, m, cw, ch, start, goal=None, mode: str = "center"):
    """带离墙惩罚的 Dijkstra。goal 为 None 时返回到"最远可达格"的路径。"""
    dist = [[INF] * m for _ in range(n)]
    prev = [[None] * m for _ in range(n)]
    dist[start[0]][start[1]] = 0.0
    heap = [(0.0, start[0], start[1])]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i][j] + 1e-12:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if not (0 <= ni < n and 0 <= nj < m) or not free[ni][nj]:
                    continue
                step = math.hypot(di * cw, dj * ch)
                pen = _wall_penalty(wall[ni][nj], mode)
                nd = d + step * pen
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    prev[ni][nj] = (i, j)
                    heapq.heappush(heap, (nd, ni, nj))

    if goal is None:
        best_d, bi, bj = -1.0, start[0], start[1]
        for i in range(n):
            for j in range(m):
                if dist[i][j] < INF and dist[i][j] > best_d:
                    best_d, bi, bj = dist[i][j], i, j
        goal = (bi, bj)
    if dist[goal[0]][goal[1]] >= INF:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur[0]][cur[1]]
    path.reverse()
    return path


def _rdp(points: Sequence[Point], eps: float) -> List[Point]:
    """Douglas–Peucker 折线简化。"""
    if len(points) < 3:
        return list(points)
    a, b = points[0], points[-1]
    dmax, idx = -1.0, 0
    ab = vsub(b, a)
    L = vdist(a, b) or 1.0
    for k in range(1, len(points) - 1):
        ap = vsub(points[k], a)
        d = abs(ab[0] * ap[1] - ab[1] * ap[0]) / L
        if d > dmax:
            dmax, idx = d, k
    if dmax <= eps:
        return [a, b]
    return _rdp(points[: idx + 1], eps)[:-1] + _rdp(points[idx:], eps)


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def compute_aisle(scene, width: Optional[float] = None,
                  mode: str = "center") -> Tuple[List[Point], List[OBB]]:
    """算出动线中心线与禁放带。返回 (polyline, band_obbs)。

    mode:
      center   —— 带离墙惩罚，通道尽量走房间中间（两侧都能摆东西，最理想）
      shortest —— 纯最短路，允许通道偏一侧（房间窄的时候靠一侧才放得下大件）
    """
    return _compute_aisle(scene, width, mode)


def _compute_aisle(scene, width: Optional[float], mode: str) -> Tuple[List[Point], List[OBB]]:
    """polyline 是世界坐标下的折线；band_obbs 是沿折线生成的矩形并集（近似带）。"""
    width = cfg.AISLE_WIDTH if width is None else width
    if width <= 0:
        return [], []

    enter = scene.enter_door
    exit_ = scene.exit_door
    a, b = enter
    start_p = vadd(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), vmul(scene.door_inward, 150.0))
    goal_p = None
    if exit_ is not None and exit_ != enter:
        c, d = exit_
        goal_p = vadd(((c[0] + d[0]) / 2.0, (c[1] + d[1]) / 2.0), vmul(scene.exit_inward, 150.0))

    g = build_grid(scene)
    if g is None:
        return [], []
    free, wall, n, m, x0, y0, cw, ch = g

    start = _nearest_free(free, n, m, x0, y0, cw, ch, start_p)
    if start is None:
        return [], []
    goal = None
    if goal_p is not None:
        goal = _nearest_free(free, n, m, x0, y0, cw, ch, goal_p)

    cells = _dijkstra(free, wall, n, m, cw, ch, start, goal, mode)
    if len(cells) < 2:
        return [], []

    pts = [(x0 + (i + 0.5) * cw, y0 + (j + 0.5) * ch) for i, j in cells]
    # 只有一个门时，通道不需要一直顶到房间最里端：最里端留给冰箱之类的大件
    if goal_p is None:
        pts = _trim_tail(pts, _end_margin(scene))
    poly = _rdp(pts, eps=max(cw, ch) * 1.2)
    if len(poly) < 2:
        poly = [pts[0], pts[-1]]

    band: List[OBB] = []
    for p, q in zip(poly, poly[1:]):
        L = vdist(p, q)
        if L < 1e-6:
            continue
        mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
        ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        band.append(OBB(mid[0], mid[1], L / 2.0, width / 2.0, ang))
    for p in poly:                      # 转角处补一个方块，避免两段之间留缺口
        band.append(OBB(p[0], p[1], width / 2.0, width / 2.0, 0.0))
    return poly, band


def _end_margin(scene) -> float:
    """通道末端预留多长：取最大件的"进深"，正好放得下冰箱这一类大件。"""
    if not scene.items:
        return 0.0
    return max(min(it.length, it.width) for it in scene.items)


def _trim_tail(pts: List[Point], margin: float) -> List[Point]:
    """从末端截掉 margin 长度（至少保留 40%），让最里端留给大件。"""
    if margin <= 0 or len(pts) < 2:
        return pts
    total = sum(vdist(p, q) for p, q in zip(pts, pts[1:]))
    keep = max(total * 0.4, total - margin)
    acc = 0.0
    out = [pts[0]]
    for p, q in zip(pts, pts[1:]):
        step = vdist(p, q)
        if acc + step <= keep:
            out.append(q)
            acc += step
        else:
            t = (keep - acc) / step if step > 1e-9 else 0.0
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
            break
    return out


def aisle_length(poly: Sequence[Point]) -> float:
    return sum(vdist(p, q) for p, q in zip(poly, poly[1:]))


def nearest_on_polyline(poly: Sequence[Point], p: Point) -> Tuple[float, Point]:
    """点到折线的最短距离与最近点。"""
    if not poly:
        return INF, p
    best_d, best_p = INF, poly[0]
    for a, b in zip(poly, poly[1:]):
        d = point_segment_distance(p, a, b)
        if d < best_d:
            best_d = d
            t = 0.0
            L = vdist(a, b)
            if L > 1e-9:
                t = min(1.0, max(0.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / (L * L)))
            best_p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    if len(poly) == 1:
        best_d, best_p = vdist(p, poly[0]), poly[0]
    return best_d, best_p


def faces_aisle(poly: Sequence[Point], face_mid: Point, outward: Point,
                max_dist: Optional[float] = None) -> bool:
    """判断某个物品的"交互面"是否朝向动线：交互面外法向指向动线，且距离不远。"""
    if not poly:
        return False
    max_dist = cfg.AISLE_FACING_MAX_DIST if max_dist is None else max_dist
    d, near = nearest_on_polyline(poly, face_mid)
    if d > max_dist:
        return False
    v = vsub(near, face_mid)
    return (v[0] * outward[0] + v[1] * outward[1]) > 0
