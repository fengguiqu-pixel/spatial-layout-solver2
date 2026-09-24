"""核心求解器。

思路（一句话）：**先把房间整体旋转到某个"墙面朝向"，在这个朝向里所有物品都是轴对齐矩形，
于是"贴墙摆放"就退化成"把矩形推到某条墙上再沿墙滑动"；对每个物品枚举候选位置、打分、
按分数从高到低做带回溯的深度优先搜索。**

关键设计
--------
1. **朝向枚举**：题目允许物体与轮廓边平行或垂直，所以合法朝向由轮廓边的方向决定。
   把所有边方向按 mod 90° 聚类，得到 1~2 个"朝向族"（例如 example1 是轴对齐族和一个
   约 15.86° 的斜族），逐族求解，最后取贴墙效果最好的那一族。
2. **候选位置生成**：对每条可贴的直墙，把矩形贴上去（外表面与墙线重合，向室内偏移），
   再沿墙滑动。滑动的候选点 = 墙两端 + 所有轮廓顶点投影 + 已放物品/禁区的边界投影 +
   等距采样点。这样既能贴角，也能紧贴已放物品，不会产生"悬空又对不齐"的丑摆放。
3. **打分**：优先贴角（同时贴两面墙）> 贴单面墙 > 悬空；同级别里接触长度越长越好；
   再同级别则离已放物品群越近越好（摆得更紧凑）。
4. **回溯**：贪心一次不成功就按打分顺序换候选重来，受节点数与时间预算限制。
5. **特殊约束**：
   * 门：门洞线段本身不允许被任何矩形压到（可用 DOOR_CLEARANCE_DEPTH 加大禁放深度）。
   * 内开门：门内侧 N×N 的方形空间直接作为禁放区。
   * 冰箱：开门边（length 所在的那条边）必须朝向室内（1mm 探测带落在轮廓内），
     并且后面摆放的物品不允许与这条边相贴/重叠。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import config as cfg
from .geometry import (OBB, Point, Rect, edge_angle_deg, norm_angle_180, obb_edge_contact_length,
                       obb_overlap, point_in_polygon, polygon_edges, ray_distance_to_boundary,
                       rect_inside_polygon, rect_rect_contact, rotate_point, segment_intersects_rect,
                       vadd, vdist, vmul, vsub, vunit)
from .scene import Item, Scene


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Wall:
    """旋转坐标系下的一条可贴直墙。"""
    kind: str          # 'h' 水平墙（y=coord） / 'v' 垂直墙（x=coord）
    coord: float
    lo: float
    hi: float
    inward: int        # +1 / -1，指向室内


@dataclass
class Frame:
    """把世界坐标整体旋转 -angle 后的工作坐标系。"""
    angle: float
    poly: List[Point]
    walls: List[Wall]
    door: Tuple[Point, Point]
    reserved: List[OBB]
    bbox: Tuple[float, float, float, float]


@dataclass
class Placement:
    item: Item
    rect: Rect                 # 工作坐标系下的矩形
    obb: OBB                   # 世界坐标下的 OBB
    angle: float               # 输出角度：相对给定 (length, width) 初始姿态旋转的角度
    wall_contacts: int
    contact_len: float
    item_contacts: int = 0
    open_side: Optional[str] = None
    strip: Optional[Rect] = None   # 冰箱开门侧的禁放探测带（工作坐标系）


@dataclass
class Solution:
    feasible: bool
    frame_angle: float
    placements: List[Placement] = field(default_factory=list)
    unplaced: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    nodes: int = 0

    @property
    def total_wall_contact(self) -> float:
        return sum(p.contact_len for p in self.placements)

    @property
    def wall_contact_count(self) -> int:
        return sum(1 for p in self.placements if p.wall_contacts > 0)

    @property
    def wall_area_ratio(self) -> float:
        """贴在墙上的物品面积占比：大件贴墙比小件贴墙更有意义，用它做主排序。"""
        total = sum(p.item.area for p in self.placements)
        if total <= 0:
            return 0.0
        return sum(p.item.area for p in self.placements if p.wall_contacts > 0) / total

    @property
    def floating_count(self) -> int:
        """既没贴墙也没贴任何已放物品的物品数量（正常应为 0）。"""
        return sum(1 for p in self.placements if p.wall_contacts == 0 and p.item_contacts == 0)


class _Budget:
    def __init__(self, nodes: int, seconds: float):
        self.limit = nodes
        self.deadline = time.time() + seconds
        self.nodes = 0
        self.exhausted = False

    def tick(self) -> bool:
        self.nodes += 1
        if self.nodes > self.limit or time.time() > self.deadline:
            self.exhausted = True
        return not self.exhausted


# ---------------------------------------------------------------------------
# 朝向枚举
# ---------------------------------------------------------------------------
def candidate_frames(scene: Scene) -> List[float]:
    """由轮廓边方向聚类出所有合法朝向族（mod 90°），按该朝向的墙总长降序返回。"""
    raw: List[Tuple[float, float]] = []
    for a, b in polygon_edges(scene.polygon):
        L = vdist(a, b)
        if L < 1.0:
            continue
        raw.append((edge_angle_deg(a, b) % 90.0, L))

    clusters: List[List[Tuple[float, float]]] = []
    for ang, L in sorted(raw):
        for c in clusters:
            d = abs(ang - c[0][0])
            d = min(d, 90.0 - d)
            if d <= cfg.FRAME_CLUSTER_TOL:
                c.append((ang, L))
                break
        else:
            clusters.append([(ang, L)])

    out: List[Tuple[float, float]] = []
    for c in clusters:
        # 周期 90° 的圆上做加权平均：用 4θ 表示，长度加权
        sx = sum(L * math.cos(math.radians(4.0 * a)) for a, L in c)
        sy = sum(L * math.sin(math.radians(4.0 * a)) for a, L in c)
        mean = (math.degrees(math.atan2(sy, sx)) / 4.0) % 90.0
        if min(mean, 90.0 - mean) <= cfg.FRAME_SNAP_TOL:
            mean = 0.0
        out.append((mean, sum(L for _, L in c)))
    out.sort(key=lambda t: -t[1])
    return [a for a, _ in out]


def build_frame(scene: Scene, angle: float, extra_reserved: Sequence[OBB] = ()) -> Frame:
    poly = [rotate_point(p, -angle) for p in scene.polygon]
    door = (rotate_point(scene.door[0], -angle), rotate_point(scene.door[1], -angle))
    reserved = [r.rotated(-angle) for r in scene.reserved]
    reserved.extend(r.rotated(-angle) for r in extra_reserved)

    walls: List[Wall] = []
    for a, b in polygon_edges(poly):
        L = vdist(a, b)
        if L < 1.0:
            continue
        dx, dy = b[0] - a[0], b[1] - a[1]
        tilt = cfg.WALL_TILT_RATIO * L
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        probe = 1.0
        if abs(dy) <= tilt and abs(dx) > 1e-9:
            inward = 1 if point_in_polygon((mid[0], mid[1] + probe), poly, 1e-6) == 1 else -1
            walls.append(Wall("h", (a[1] + b[1]) / 2.0, min(a[0], b[0]), max(a[0], b[0]), inward))
        elif abs(dx) <= tilt and abs(dy) > 1e-9:
            inward = 1 if point_in_polygon((mid[0] + probe, mid[1]), poly, 1e-6) == 1 else -1
            walls.append(Wall("v", (a[0] + b[0]) / 2.0, min(a[1], b[1]), max(a[1], b[1]), inward))

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Frame(angle=angle, poly=poly, walls=walls, door=door, reserved=reserved,
                 bbox=(min(xs), min(ys), max(xs), max(ys)))


# ---------------------------------------------------------------------------
# 候选位置生成
# ---------------------------------------------------------------------------
def _axis_positions(lo: float, hi: float, size: float, marks: Sequence[float], step: float) -> List[float]:
    """沿墙方向的候选起点（矩形在该轴上靠近 lo 的那一端坐标）。"""
    starts = {lo, hi - size}
    for m in marks:
        starts.add(m)
        starts.add(m - size)
    span = (hi - lo) - size
    if span > 0:
        n = max(1, int(span / step))
        d = span / n
        starts.update(lo + i * d for i in range(n + 1))
    else:
        starts.add(lo + span / 2.0)
    return sorted(v for v in starts if lo - size - 1e-9 <= v <= hi + 1e-9)


def _obb_to_rect(obb: OBB) -> Rect:
    """禁放区（OBB）在工作坐标系下应当也是轴对齐的，转成 Rect 便于统一处理。"""
    a = math.fmod(obb.angle, 180.0)
    hw, hh = (obb.hw, obb.hh) if abs(a) < 1e-6 or abs(abs(a) - 180.0) < 1e-6 else (obb.hh, obb.hw)
    return Rect(obb.cx - hw, obb.cy - hh, obb.cx + hw, obb.cy + hh)


def reserved_rects(frame: Frame) -> List[Rect]:
    return [_obb_to_rect(o) for o in frame.reserved]


def _rect_faces(r: Rect) -> List[Wall]:
    """已放物品（轴对齐矩形）的四条边都可以当成"可贴的面"，
    这样才能铺第二排、第三排，而不是只能沿墙铺一排。"""
    return [
        Wall("v", r.x0, r.y0, r.y1, -1),
        Wall("v", r.x1, r.y0, r.y1, +1),
        Wall("h", r.y0, r.x0, r.x1, -1),
        Wall("h", r.y1, r.x0, r.x1, +1),
    ]


def contact_faces(frame: Frame, placed: Sequence[Rect], forbidden: Sequence[Rect]) -> List[Wall]:
    """所有可贴的面：真实墙 + 已放物品的边 + 禁放区的边。"""
    faces: List[Wall] = list(frame.walls)
    for r in placed:
        faces.extend(_rect_faces(r))
    for r in forbidden:
        faces.extend(_rect_faces(r))
    return faces


def gen_positions(frame: Frame, w: float, h: float, placed: Sequence[Rect],
                  forbidden: Sequence[Rect]) -> List[Rect]:
    """枚举"贴在某个面上"的所有候选矩形（工作坐标系，轴对齐）。

    贴的面包括真实墙面和已放物品的边；滑动的候选点取面两端、所有顶点/已放物品边界
    的投影，再加等距采样，这样既能贴角、也能紧贴已放物品。
    """
    marks_x: List[float] = [p[0] for p in frame.poly]
    marks_y: List[float] = [p[1] for p in frame.poly]
    for r in placed:
        marks_x.extend((r.x0, r.x1))
        marks_y.extend((r.y0, r.y1))
    for r in forbidden:
        marks_x.extend((r.x0, r.x1))
        marks_y.extend((r.y0, r.y1))

    out: List[Rect] = []
    seen = set()
    for wall in contact_faces(frame, placed, forbidden):
        if wall.kind == "h":
            y0 = wall.coord if wall.inward > 0 else wall.coord - h
            for x0 in _axis_positions(wall.lo, wall.hi, w, marks_x, cfg.SLIDE_STEP):
                key = (round(x0, 2), round(y0, 2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Rect(x0, y0, x0 + w, y0 + h))
        else:
            x0 = wall.coord if wall.inward > 0 else wall.coord - w
            for y0 in _axis_positions(wall.lo, wall.hi, h, marks_y, cfg.SLIDE_STEP):
                key = (round(x0, 2), round(y0, 2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Rect(x0, y0, x0 + w, y0 + h))
    return out


def rect_valid(frame: Frame, rect: Rect, placed: Sequence[Rect], forbidden: Sequence[Rect],
               strips: Sequence[Rect] = ()) -> bool:
    """矩形是否合法：在轮廓内、不与已放物品/禁区重叠、不压门洞。"""
    obb = rect.as_obb()
    if not rect_inside_polygon(obb, frame.poly, cfg.INSIDE_TOL):
        return False
    for r in placed:
        if obb_overlap(obb, r.as_obb(), cfg.OVERLAP_TOL):
            return False
    for r in forbidden:
        if obb_overlap(obb, r.as_obb(), cfg.OVERLAP_TOL):
            return False
    for r in strips:
        if obb_overlap(obb, r.as_obb(), cfg.STRIP_TOL):
            return False
    if segment_intersects_rect(frame.door[0], frame.door[1], obb, 0.0):
        return False
    return True


def _cheap_reject(frame: Frame, rect: Rect) -> bool:
    """便宜的预筛：连轮廓包围盒都不相交 / 中心点都不在轮廓内的直接扔掉。"""
    x0, y0, x1, y1 = frame.bbox
    if rect.x1 < x0 or rect.x0 > x1 or rect.y1 < y0 or rect.y0 > y1:
        return True
    return point_in_polygon(rect.center, frame.poly, 1e-6) < 0


def _score(frame: Frame, rect: Rect, placed: Sequence[Rect]) -> Tuple[float, float, float]:
    """候选打分（越小越好）：贴合级别 → 接触长度 → 与已放物品的紧凑度。

    贴合级别：贴墙角(0) > 贴一面墙(1) > 贴墙又贴物品(1.5) > 只贴物品(2) > 悬空(5)
    """
    obb = rect.as_obb()
    wall_contacts = 0
    wall_len = 0.0
    for a, b in polygon_edges(frame.poly):
        L = obb_edge_contact_length(obb, a, b, cfg.CONTACT_TOL)
        if L > cfg.CONTACT_TOL:
            wall_contacts += 1
            wall_len += L
    item_contacts = 0
    item_len = 0.0
    for r in placed:
        L = rect_rect_contact(rect, r, cfg.CONTACT_TOL)
        if L > cfg.CONTACT_TOL:
            item_contacts += 1
            item_len += L
    if wall_contacts >= 2:
        penalty = 0.0
    elif wall_contacts == 1:
        penalty = 1.0 if item_contacts == 0 else 1.5
    else:
        penalty = 2.0 if item_contacts >= 1 else 5.0
    if placed:
        cx = sum((r.x0 + r.x1) / 2.0 for r in placed) / len(placed)
        cy = sum((r.y0 + r.y1) / 2.0 for r in placed) / len(placed)
        dist = math.hypot(rect.center[0] - cx, rect.center[1] - cy)
    else:
        dist = 0.0
    return (penalty, -(wall_len + 0.5 * item_len), dist)


def _wall_contact_info(frame: Frame, rect: Rect) -> Tuple[int, float]:
    obb = rect.as_obb()
    n = 0
    total = 0.0
    for a, b in polygon_edges(frame.poly):
        L = obb_edge_contact_length(obb, a, b, cfg.CONTACT_TOL)
        if L > cfg.CONTACT_TOL:
            n += 1
            total += L
    return n, total


def _item_contact_count(rect: Rect, placed: Sequence[Rect]) -> int:
    return sum(1 for r in placed if rect_rect_contact(rect, r, cfg.CONTACT_TOL) > cfg.CONTACT_TOL)


def ranked_candidates(frame: Frame, item: Item, placed: Sequence[Rect],
                      forbidden: Sequence[Rect], strips: Sequence[Rect] = ()
                      ) -> List[Tuple[Tuple[float, float, float], Rect]]:
    out: List[Tuple[Tuple[float, float, float], Rect]] = []
    seen = set()
    for w, h in ((item.length, item.width), (item.width, item.length)):
        for rect in gen_positions(frame, w, h, placed, forbidden):
            key = (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))
            if key in seen:
                continue
            seen.add(key)
            if _cheap_reject(frame, rect):
                continue
            if not rect_valid(frame, rect, placed, forbidden, strips):
                continue
            out.append((_score(frame, rect, placed), rect))
    out.sort(key=lambda t: t[0])
    if len(out) > cfg.MAX_CANDIDATES:
        stride = len(out) / float(cfg.MAX_CANDIDATES)
        out = [out[int(i * stride)] for i in range(cfg.MAX_CANDIDATES)]
    return out


# ---------------------------------------------------------------------------
# 冰箱开门侧
# ---------------------------------------------------------------------------
def fridge_open_strip(frame: Frame, rect: Rect, item: Item, placed: Sequence[Rect],
                      forbidden: Sequence[Rect], strips: Sequence[Rect] = ()
                      ) -> Optional[Tuple[str, Rect]]:
    """挑一条朝向室内的 length 边作为开门边，返回 (边名, 禁放探测带)。"""
    buf = cfg.FRIDGE_OPEN_BUFFER
    eps = 1e-6
    options: List[Tuple[str, Rect, Point]] = []
    if abs(rect.w - item.length) < eps:          # length 沿 x → 上下两条边是 length 边
        options.append(("bottom", Rect(rect.x0, rect.y0 - buf, rect.x1, rect.y0), (0.0, -1.0)))
        options.append(("top", Rect(rect.x0, rect.y1, rect.x1, rect.y1 + buf), (0.0, 1.0)))
    else:                                         # length 沿 y → 左右两条边是 length 边
        options.append(("left", Rect(rect.x0 - buf, rect.y0, rect.x0, rect.y1), (-1.0, 0.0)))
        options.append(("right", Rect(rect.x1, rect.y0, rect.x1 + buf, rect.y1), (1.0, 0.0)))

    best: Optional[Tuple[float, str, Rect]] = None
    for side, strip, d in options:
        obb = strip.as_obb()
        if not rect_inside_polygon(obb, frame.poly, 1e-6):
            continue                              # 这一侧顶着墙，门打不开
        if any(obb_overlap(obb, r.as_obb(), 0.0) for r in placed):
            continue
        if any(obb_overlap(obb, r.as_obb(), 0.0) for r in forbidden):
            continue
        if any(obb_overlap(obb, r.as_obb(), 0.0) for r in strips):
            continue
        if segment_intersects_rect(frame.door[0], frame.door[1], obb, 0.0):
            continue
        face = (rect.center[0] + d[0] * rect.w / 2.0, rect.center[1] + d[1] * rect.h / 2.0)
        free = ray_distance_to_boundary(frame.poly, face, d, maxd=10000.0)
        if best is None or free > best[0]:
            best = (free, side, strip)
    if best is None:
        return None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------
def _to_world(frame: Frame, rect: Rect) -> OBB:
    c = rotate_point(rect.center, frame.angle)
    return OBB(c[0], c[1], rect.w / 2.0, rect.h / 2.0, frame.angle)


def _output_angle(frame: Frame, rect: Rect, item: Item) -> float:
    """相对给定 (length, width) 初始姿态（0°）旋转的角度。"""
    if abs(rect.w - item.length) < 1e-6:
        return norm_angle_180(frame.angle)
    return norm_angle_180(frame.angle + 90.0)


def _dfs(frame: Frame, order: Sequence[Item], idx: int, placed: List[Placement],
         zones: List[Rect], strips: List[Rect], budget: _Budget,
         allow_floating: bool = True) -> Optional[List[Placement]]:
    if idx >= len(order):
        return list(placed)
    if not budget.tick():
        return None

    item = order[idx]
    placed_rects = [p.rect for p in placed]
    cands = ranked_candidates(frame, item, placed_rects, zones, strips)
    if not allow_floating:
        cands = [c for c in cands if c[0][0] < 5.0]

    for _, rect in cands[: cfg.MAX_BRANCH]:
        strip = None
        if item.is_fridge:
            got = fridge_open_strip(frame, rect, item, placed_rects, zones, strips)
            if got is None:
                continue
            side, strip = got
        else:
            side = None

        wc, clen = _wall_contact_info(frame, rect)
        ic = _item_contact_count(rect, placed_rects)
        pl = Placement(item=item, rect=rect, obb=_to_world(frame, rect),
                       angle=_output_angle(frame, rect, item), wall_contacts=wc,
                       contact_len=clen, item_contacts=ic, open_side=side, strip=strip)
        placed.append(pl)
        if strip is not None:
            strips.append(strip)
        res = _dfs(frame, order, idx + 1, placed, zones, strips, budget, allow_floating)
        if res is not None:
            return res
        if strip is not None:
            strips.pop()
        placed.pop()
        if budget.exhausted:
            return None
    return None


def _solve_frame(scene: Scene, angle: float, extra_reserved: Sequence[OBB] = ()) -> Optional[Solution]:
    frame = build_frame(scene, angle, extra_reserved)

    # 摆放顺序：面积降序 / 最长边降序 / 面积升序，三种都试，取贴墙效果最好的
    orders = [
        sorted(scene.items, key=lambda it: (-it.area, it.name)),
        sorted(scene.items, key=lambda it: (-max(it.length, it.width), -it.area, it.name)),
        sorted(scene.items, key=lambda it: (it.area, it.name)),
    ]
    best: Optional[Solution] = None
    nodes = 0
    # 第一轮：不允许悬空（必须贴墙或贴已放物品）—— 对应题目"优先考虑贴墙"
    # 第二轮：放不下再放宽。每个顺序单独计时，避免某一次搜索把预算烧光
    for allow_floating in (False, True):
        for order in orders:
            seconds = cfg.TIME_BUDGET * (0.25 if not allow_floating else 0.5)
            budget = _Budget(cfg.NODE_BUDGET, seconds)
            res = _dfs(frame, order, 0, [], reserved_rects(frame), [], budget, allow_floating)
            nodes += budget.nodes
            if res:
                sol = Solution(feasible=True, frame_angle=angle, placements=res, nodes=nodes)
                key = (sol.wall_area_ratio, sol.wall_contact_count, sol.total_wall_contact)
                best_key = None if best is None else (
                    best.wall_area_ratio, best.wall_contact_count, best.total_wall_contact)
                if best is None or key > best_key:
                    best = sol
        if best is not None:
            break
    return best


def solve(scene: Scene, door_clearance: Optional[float] = None) -> Solution:
    """求解一个场景。返回 Solution（feasible=False 表示放不下）。"""
    t0 = time.time()
    extra: List[OBB] = []
    depth = cfg.DOOR_CLEARANCE_DEPTH if door_clearance is None else door_clearance
    if depth > 0:
        a, b = scene.door
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        center = vadd(mid, vmul(scene.door_inward, depth / 2.0))
        extra.append(OBB(center[0], center[1], scene.door_width / 2.0, depth / 2.0,
                         edge_angle_deg(a, b)))

    frames = candidate_frames(scene)
    best: Optional[Solution] = None
    for ang in frames:
        sol = _solve_frame(scene, ang, extra)
        if sol is not None and sol.feasible:
            key = (sol.wall_area_ratio, sol.wall_contact_count, sol.total_wall_contact)
            best_key = None if best is None else (
                best.wall_area_ratio, best.wall_contact_count, best.total_wall_contact)
            if best is None or key > best_key:
                best = sol
    if best is None:
        return Solution(feasible=False, frame_angle=frames[0] if frames else 0.0,
                        unplaced=[it.name for it in scene.items], elapsed=time.time() - t0)
    best.elapsed = time.time() - t0
    return best


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def solution_to_dict(scene: Scene, sol: Solution) -> Dict:
    items = []
    for p in sol.placements:
        d = {
            "name": p.item.name,
            "type": p.item.kind,
            "center": [round(p.obb.cx, 2), round(p.obb.cy, 2)],
            "angle": round(p.angle, 3),
            "size": [p.item.length, p.item.width],
            "wall_contacts": p.wall_contacts,
        }
        if p.open_side:
            d["fridge_opening_side"] = p.open_side
        items.append(d)
    return {
        "feasible": sol.feasible,
        "frame_angle": round(sol.frame_angle, 3),
        "items_total": len(scene.items),
        "items_placed": len(sol.placements),
        "wall_hugging": f"{sol.wall_contact_count}/{len(scene.items)}",
        "wall_area_ratio": round(sol.wall_area_ratio, 3),
        "floating": sol.floating_count,
        "unplaced": sol.unplaced,
        "elapsed_sec": round(sol.elapsed, 3),
        "placements": items,
    }
