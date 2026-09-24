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
from .circulation import aisle_length, faces_aisle, nearest_on_polyline
from .geometry import (OBB, Point, Rect, edge_angle_deg, norm_angle_180, obb_edge_contact_length,
                       obb_overlap, point_in_polygon, polygon_area, polygon_bbox, polygon_edges,
                       ray_distance_to_boundary, rect_inside_polygon, rect_rect_contact,
                       _proj_range,
                       rotate_point, segment_intersects_rect, vadd, vdist, vdot, vlen, vmul,
                       vsub, vunit)
from .scene import Item, Scene, build_aisle


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
    reserved: List[OBB]                       # 内开门 N×N（旋转后）
    zones_obb: List[OBB] = field(default_factory=list)   # 全部硬禁放区（内开门 + 动线带）
    face_rects: List[Rect] = field(default_factory=list)  # 可作为贴面的禁放区（轴对齐的）
    aisle_rects: List[Rect] = field(default_factory=list)  # 通道带中轴对齐的段，也可贴着摆
    aisle_poly: List[Point] = field(default_factory=list)  # 动线中心线（世界坐标）
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


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
    facing_aisle: bool = False     # 交互面（冰箱开门边 / 货架 length 边）是否朝向动线


@dataclass
class Solution:
    feasible: bool
    frame_angle: float
    placements: List[Placement] = field(default_factory=list)
    unplaced: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    nodes: int = 0
    reason: str = ""          # 判定不可行时给的原因（快速预检命中时才有）
    aisle_width: float = 0.0  # 实际采用的动线宽度（0 表示没留通道）
    wall_usage: float = 0.0   # 墙边利用率：被物品占用的墙长 / 可贴墙总长

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
    def aisle_facing_count(self) -> int:
        return sum(1 for p in self.placements if p.facing_aisle)

    @property
    def floating_count(self) -> int:
        """既没贴墙也没贴任何已放物品的物品数量（正常应为 0）。"""
        return sum(1 for p in self.placements if p.wall_contacts == 0 and p.item_contacts == 0)


def _spread(sol: "Solution") -> float:
    """物品中心点到它们质心的平均距离：越小说明摆得越集中（剩余空地更容易连成整块）。"""
    if len(sol.placements) < 2:
        return 0.0
    n = len(sol.placements)
    cx = sum(p.obb.cx for p in sol.placements) / n
    cy = sum(p.obb.cy for p in sol.placements) / n
    return sum(math.hypot(p.obb.cx - cx, p.obb.cy - cy) for p in sol.placements) / n


def _rank_key(sol: "Solution") -> Tuple:
    """方案排序键（越大越好）。

    贴墙模式：先看**墙边利用率**（墙被占用了多少，直接对应"墙边有没有被浪费"），
    再看贴墙面积占比、贴墙件数、接触长度；紧凑模式看集中度。
    """
    if cfg.COMPACT_MODE:
        return (-_spread(sol), sol.wall_usage, sol.wall_area_ratio,
                sol.wall_contact_count, sol.total_wall_contact)
    return (sol.wall_usage, sol.wall_area_ratio, sol.wall_contact_count, sol.total_wall_contact)


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
    enter = scene.enter_door
    door = (rotate_point(enter[0], -angle), rotate_point(enter[1], -angle))
    reserved = [r.rotated(-angle) for r in scene.reserved]
    reserved.extend(r.rotated(-angle) for r in extra_reserved)
    # 硬禁放区 = 内开门 N×N + 动线通道带（都不允许物品压入）
    zones = [r.rotated(-angle) for r in scene.zones]
    zones.extend(r.rotated(-angle) for r in extra_reserved)
    # 只有轴对齐的禁放区（内开门方形）才拿来当"可贴的面"，
    # 动线带不参与贴面，免得把物品摆到通道边上悬空
    face_rects = [_obb_to_rect(o) for o in reserved if _is_axis_aligned(o)]
    # 通道带里与当前朝向平行的段也可以贴着摆（货架沿通道排是正常做法）
    aisle_rects = [_obb_to_rect(o) for o in scene.aisle_band
                   if _is_axis_aligned(o.rotated(-angle))]
    aisle_rects = [_obb_to_rect(o.rotated(-angle)) for o in scene.aisle_band
                   if _is_axis_aligned(o.rotated(-angle))]

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
                 zones_obb=zones, face_rects=face_rects, aisle_rects=aisle_rects,
                 aisle_poly=list(scene.aisle_poly),
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


def _is_axis_aligned(obb: OBB, tol: float = 1e-6) -> bool:
    a = math.fmod(obb.angle, 90.0)
    return abs(a) <= tol or abs(abs(a) - 90.0) <= tol


def _obb_to_rect(obb: OBB) -> Rect:
    """禁放区（OBB）在工作坐标系下应当也是轴对齐的，转成 Rect 便于统一处理。"""
    a = math.fmod(obb.angle, 180.0)
    hw, hh = (obb.hw, obb.hh) if abs(a) < 1e-6 or abs(abs(a) - 180.0) < 1e-6 else (obb.hh, obb.hw)
    return Rect(obb.cx - hw, obb.cy - hh, obb.cx + hw, obb.cy + hh)


def reserved_rects(frame: Frame) -> List[Rect]:
    return list(frame.face_rects)


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
    for z in frame.zones_obb:          # 内开门 N×N + 动线通道
        if obb_overlap(obb, z, cfg.OVERLAP_TOL):
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


def _wall_contact_segments(frame: Frame, rect: Rect) -> List[Tuple[object, float, float]]:
    """矩形贴到了哪些墙段，返回 [(墙标识, 起点, 终点)]（沿墙方向的投影区间）。"""
    obb = rect.as_obb()
    out: List[Tuple[object, float, float]] = []
    for a, b in polygon_edges(frame.poly):
        if obb_edge_contact_length(obb, a, b, cfg.CONTACT_TOL) <= cfg.CONTACT_TOL:
            continue
        d = vsub(b, a)
        L = vlen(d)
        if L <= 1e-9:
            continue
        u = (d[0] / L, d[1] / L)
        proj = [vdot(vsub(c, a), u) for c in obb.corners()]
        lo, hi = min(proj), max(proj)
        lo = max(0.0, min(lo, L))
        hi = max(0.0, min(hi, L))
        if hi - lo > cfg.CONTACT_TOL:
            key = ("e", round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1))
            out.append((key, lo, hi))
    return out


def _union_len(segs: Sequence[Tuple[float, float]]) -> float:
    if not segs:
        return 0.0
    ss = sorted(segs)
    total = 0.0
    cs, ce = ss[0]
    for a, b in ss[1:]:
        if a > ce:
            total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    total += ce - cs
    return total


def _cover_gain(cov: Dict[object, List[Tuple[float, float]]],
                segs: Sequence[Tuple[object, float, float]]) -> float:
    """这些墙段能带来多少"以前没被占用的新墙长"。"""
    gain = 0.0
    for key, lo, hi in segs:
        cur = cov.get(key, [])
        gain += _union_len(list(cur) + [(lo, hi)]) - _union_len(cur)
    return gain


def _aisle_facing(frame: Frame, rect: Rect, item: Item) -> bool:
    """物品的交互面（冰箱开门边 / 其它物品的 length 边）是否朝向动线。"""
    if not frame.aisle_poly:
        return False
    eps = 1e-6
    if abs(rect.w - item.length) < eps:      # length 沿 x → 上下两面是 length 边
        cands = [((rect.center[0], rect.y1), (0.0, 1.0)),
                 ((rect.center[0], rect.y0), (0.0, -1.0))]
    else:                                    # length 沿 y → 左右两面是 length 边
        cands = [((rect.x1, rect.center[1]), (1.0, 0.0)),
                 ((rect.x0, rect.center[1]), (-1.0, 0.0))]
    for mid, n in cands:
        wmid = rotate_point(mid, frame.angle)
        wn = rotate_point((n[0], n[1]), frame.angle)
        if faces_aisle(frame.aisle_poly, wmid, wn):
            return True
    return False


def _score(frame: Frame, rect: Rect, placed: Sequence[Rect], item: Item,
           cov: Optional[Dict[object, List[Tuple[float, float]]]] = None) -> Tuple:
    """候选打分（越小越好）。

    量级依次是：
      贴合级别(0~5) → 是否朝向动线(0/1) → **新占用的墙长**（越大越好）
      → 接触长度（贴得紧） → 离已放物品群的距离（越近越紧凑，权重最低）

    "新占用的墙长"这一项是让墙边被充分利用的关键：以前只奖励"贴墙"，
    物品会扎堆在已放物品旁边，剩下的墙段白白空着；现在奖励去占还没人用的墙段。
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
    aisle_contacts = sum(1 for r in frame.aisle_rects
                         if rect_rect_contact(rect, r, cfg.CONTACT_TOL) > cfg.CONTACT_TOL)
    if wall_contacts >= 2:
        penalty = 0.0
    elif wall_contacts == 1:
        penalty = 1.0 if (item_contacts == 0 and aisle_contacts == 0) else 1.5
    else:
        penalty = 2.0 if (item_contacts >= 1 or aisle_contacts >= 1) else 5.0

    new_wall = 0.0
    if cov is not None:
        new_wall = _cover_gain(cov, _wall_contact_segments(frame, rect))

    facing = 0.0 if _aisle_facing(frame, rect, item) else 1.0

    dist = _dist_to_cluster(rect, placed)
    if cfg.COMPACT_MODE:
        return (penalty, -new_wall, facing, dist, -(wall_len + 0.5 * item_len))
    # 贴墙级别 → 新占用墙长 → 是否朝向动线 → 接触长度 → 紧凑度
    return (penalty, -new_wall, facing, -(wall_len + 0.5 * item_len), dist)


def _dist_to_cluster(rect: Rect, placed: Sequence[Rect]) -> float:
    if not placed:
        return 0.0
    cx = sum((r.x0 + r.x1) / 2.0 for r in placed) / len(placed)
    cy = sum((r.y0 + r.y1) / 2.0 for r in placed) / len(placed)
    return math.hypot(rect.center[0] - cx, rect.center[1] - cy)


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
                      forbidden: Sequence[Rect], strips: Sequence[Rect] = (),
                      cov: Optional[Dict[object, List[Tuple[float, float]]]] = None
                      ) -> List[Tuple[Tuple, Rect]]:
    out: List[Tuple[Tuple, Rect]] = []
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
            out.append((_score(frame, rect, placed, item, cov), rect))
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
        # 优先选"朝着动线"的那一面，其次选前方空间大的
        wface = rotate_point(face, frame.angle)
        wn = rotate_point((d[0], d[1]), frame.angle)
        to_aisle = 1 if faces_aisle(frame.aisle_poly, wface, wn) else 0
        key = (to_aisle, free)
        if best is None or key > best[0]:
            best = (key, side, strip)
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
         allow_floating: bool = True,
         cov: Optional[Dict[object, List[Tuple[float, float]]]] = None
         ) -> Optional[List[Placement]]:
    if idx >= len(order):
        return list(placed)
    if not budget.tick():
        return None
    if cov is None:
        cov = {}

    item = order[idx]
    placed_rects = [p.rect for p in placed]
    cands = ranked_candidates(frame, item, placed_rects, zones, strips, cov)
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
                       contact_len=clen, item_contacts=ic, open_side=side, strip=strip,
                       facing_aisle=_aisle_facing(frame, rect, item))
        placed.append(pl)
        if strip is not None:
            strips.append(strip)
        # 记录这次新占用的墙段（回溯时要撤掉）
        added = _wall_contact_segments(frame, rect)
        for key, lo, hi in added:
            cov.setdefault(key, []).append((lo, hi))
        res = _dfs(frame, order, idx + 1, placed, zones, strips, budget, allow_floating, cov)
        if res is not None:
            return res
        for key, lo, hi in added:
            try:
                cov[key].remove((lo, hi))
            except (KeyError, ValueError):
                pass
        if strip is not None:
            strips.pop()
        placed.pop()
        if budget.exhausted:
            return None
    return None


def _solve_frame(scene: Scene, angle: float, extra_reserved: Sequence[OBB] = (),
                 scale: float = 1.0) -> Optional[Solution]:
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
        # 放宽阶段只试前两个顺序：第一遍都失败说明是硬骨头，多试一个顺序收益很低
        for order in (orders if not allow_floating else orders[:2]):
            seconds = cfg.TIME_BUDGET * (0.25 if not allow_floating else 0.5) * scale
            budget = _Budget(cfg.NODE_BUDGET, seconds)
            res = _dfs(frame, order, 0, [], reserved_rects(frame), [], budget, allow_floating)
            nodes += budget.nodes
            if res:
                sol = Solution(feasible=True, frame_angle=angle, placements=res, nodes=nodes)
                used, usable = wall_usage(scene, sol)
                sol.wall_usage = used / usable if usable > 0 else 0.0
                if best is None or _rank_key(sol) > _rank_key(best):
                    best = sol
        if best is not None:
            break
    return best


def quick_infeasibility(scene: Scene) -> str:
    """秒级的不可行预检，避免在注定放不下的输入上白白烧搜索预算。

    只做必要条件判断（面积 / 尺寸），判不出来就返回空串交给搜索去试。
    """
    room = polygon_area(scene.polygon)
    reserved = sum(r.area() for r in scene.reserved) + scene.aisle_area_est
    need = sum(it.area for it in scene.items)
    if need > room - reserved + 1e-6:
        return (f"面积不够：物品占地合计 {need:,.0f} > 可用面积 {room - reserved:,.0f} "
                f"(房间 {room:,.0f} - 禁放区 {sum(r.area() for r in scene.reserved):,.0f}"
                f" - 通道 {scene.aisle_area_est:,.0f})")
    x0, y0, x1, y1 = polygon_bbox(scene.polygon)
    bw, bh = x1 - x0, y1 - y0
    for it in scene.items:
        if min(it.length, it.width) > min(bw, bh) + 1e-6:
            return (f"{it.name} 的最小边 {min(it.length, it.width):,.0f} "
                    f"超过房间最窄处 {min(bw, bh):,.0f}")
    return ""


def _solve_once(scene: Scene, door_clearance: Optional[float], t0: float,
                scale: float = 1.0) -> Solution:
    """在当前动线宽度下完整求解一次。"""
    reason = quick_infeasibility(scene)
    if reason:
        return Solution(feasible=False, frame_angle=0.0,
                        unplaced=[it.name for it in scene.items],
                        elapsed=time.time() - t0, reason=reason)

    extra: List[OBB] = []
    depth = cfg.DOOR_CLEARANCE_DEPTH if door_clearance is None else door_clearance
    if depth > 0:
        a, b = scene.enter_door
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        center = vadd(mid, vmul(scene.door_inward, depth / 2.0))
        extra.append(OBB(center[0], center[1], vdist(a, b) / 2.0, depth / 2.0,
                         edge_angle_deg(a, b)))

    frames = candidate_frames(scene)
    best: Optional[Solution] = None
    for ang in frames:
        sol = _solve_frame(scene, ang, extra, scale)
        if sol is not None and sol.feasible:
            if best is None or _rank_key(sol) > _rank_key(best):
                best = sol
    if best is None:
        room = polygon_area(scene.polygon)
        reserved = sum(r.area() for r in scene.reserved) + sum(r.area() for r in extra)
        need = sum(it.area for it in scene.items)
        usable = max(room - reserved, 1.0)
        return Solution(feasible=False, frame_angle=frames[0] if frames else 0.0,
                        unplaced=[it.name for it in scene.items], elapsed=time.time() - t0,
                        reason=(f"搜索预算内没找到可行解：物品需求面积 {need:,.0f}，"
                                f"可用面积 {usable:,.0f}，需求密度 {need / usable * 100:.0f}%"))
    best.elapsed = time.time() - t0
    best.aisle_width = scene.aisle_width
    return best


def solve(scene: Scene, door_clearance: Optional[float] = None,
          aisle_width: Optional[float] = None) -> Solution:
    """求解一个场景。返回 Solution（feasible=False 表示放不下）。

    动线是硬约束，但宽度自适应：先按 AISLE_WIDTH 试，放不下就逐级收窄到 70% / 45%，
    最后仍放不下才取消通道——宁可通道窄一点，也不要因为留通道而整题无解。
    """
    t0 = time.time()
    if not cfg.ENABLE_AISLE:
        probes = [(0.0, "shortest")]
    elif aisle_width is not None:
        probes = [(aisle_width, m) for m in cfg.AISLE_MODES]
    else:
        w = cfg.AISLE_WIDTH
        # 先试标准宽度（居中走法最好看），再逐级收窄；收窄后只用最短路走法省时间
        probes = [(w, "center"), (w, "shortest")]
        for f in (0.8, 0.62, 0.45):
            probes.append((round(w * f), "shortest"))
        probes.append((0.0, "shortest"))

    last: Optional[Solution] = None
    for i, (w, mode) in enumerate(probes):
        build_aisle(scene, w, mode)
        # 探测阶段只给 35% 预算，快速判断这个宽度的通道能不能放下
        sol = _solve_once(scene, door_clearance, t0, scale=0.35)
        sol.aisle_width = scene.aisle_width
        if sol.feasible:
            # 找到了能放下的最宽通道，再用完整预算重跑一次拿到更好的摆法
            refined = _solve_once(scene, door_clearance, t0, scale=1.0)
            if refined.feasible and _rank_key(refined) >= _rank_key(sol):
                sol = refined
                sol.aisle_width = scene.aisle_width
            repair_overlaps(scene, sol)
            if w > 0 and w < cfg.AISLE_WIDTH:
                sol.reason = (f"房间尺寸所限，{cfg.AISLE_WIDTH:.0f} 宽的通道放不下全部物品，"
                              f"已自动收窄到 {w:.0f}（这是本房间能容纳的最宽通道）")
            elif w == 0 and cfg.ENABLE_AISLE:
                sol.reason = (f"空间不足以保留 {cfg.AISLE_WIDTH:.0f} 宽的通道，"
                              f"已取消通道约束")
            elif mode != "center":
                sol.reason = (f"通道走法：{mode}（房间偏窄，居中通道会占用摆放大件的空间）")
            return sol
        last = sol
    return last if last is not None else Solution(feasible=False, elapsed=time.time() - t0)


# ---------------------------------------------------------------------------
# 空间利用率等指标
# ---------------------------------------------------------------------------
def usable_wall_length(scene: Scene, frame_angle: float) -> float:
    """该朝向下真正可以贴的墙的总长度（与朝向平行或垂直的边）。"""
    total = 0.0
    for a, b in polygon_edges(scene.polygon):
        L = vdist(a, b)
        if L < 1.0:
            continue
        d = abs((edge_angle_deg(a, b) - frame_angle) % 90.0)
        if min(d, 90.0 - d) <= cfg.FRAME_CLUSTER_TOL:
            total += L
    return total


def free_space_grid(scene: Scene, sol: Solution, target_cells: int = None):
    """把地面打成栅格，标出哪些格子是空的。

    返回 (grid, n, m, cell_area)，grid 为一维 list，True 表示"在轮廓内且没被占用"。
    顺带做一次连通域分析，这样能区分"一大块完整空地"和"到处都是碎片"——
    后者面积虽大却没法利用，这也是判断摆放好坏的关键。
    """
    if target_cells is None:
        target_cells = cfg.GRID_CELLS
    x0, y0, x1, y1 = polygon_bbox(scene.polygon)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return [], 0, 0, 0.0
    n = max(8, int(round(math.sqrt(target_cells * w / h))))
    m = max(8, int(round(target_cells / n)))
    cw, ch = w / n, h / m
    cell = cw * ch

    blocks = [p.obb for p in sol.placements] + list(scene.reserved)
    corners = [o.corners() for o in blocks]
    bb = []
    for o in blocks:
        r = max(o.hw, o.hh) * 1.5
        bb.append((o.cx - r, o.cy - r, o.cx + r, o.cy + r))

    grid = [False] * (n * m)
    for i in range(n):
        x = x0 + (i + 0.5) * cw
        for j in range(m):
            y = y0 + (j + 0.5) * ch
            if point_in_polygon((x, y), scene.polygon, 1e-6) != 1:
                continue
            hit = False
            for k in range(len(blocks)):
                b0, b1, b2, b3 = bb[k]
                if b0 <= x <= b2 and b1 <= y <= b3 and point_in_polygon((x, y), corners[k], 1e-6) >= 0:
                    hit = True
                    break
            if not hit:
                grid[i * m + j] = True
    return grid, n, m, cell


def _largest_free_block(grid, n, m, cell: float) -> Tuple[float, int]:
    """连通域分析：返回 (最大一块空地面积, 空地碎片块数)。"""
    seen = [False] * (n * m)
    best = 0
    blocks = 0
    stack: List[int] = []
    for start in range(n * m):
        if not grid[start] or seen[start]:
            continue
        blocks += 1
        size = 0
        stack.append(start)
        seen[start] = True
        while stack:
            idx = stack.pop()
            size += 1
            i, j = divmod(idx, m)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    nidx = ni * m + nj
                    if grid[nidx] and not seen[nidx]:
                        seen[nidx] = True
                        stack.append(nidx)
        best = max(best, size)
    return best * cell, blocks


def estimate_free_area(scene: Scene, sol: Solution, target_cells: int = None) -> float:
    """栅格采样估算"还没被占用的地面面积"（不含禁放区）。"""
    grid, n, m, cell = free_space_grid(scene, sol, target_cells)
    return sum(1 for v in grid if v) * cell


def wall_usage(scene: Scene, sol: Solution) -> Tuple[float, float]:
    """(被物品占用的墙长, 该朝向下可贴墙的总长)——墙边利用率 = 两者相除。"""
    tol = cfg.CONTACT_TOL
    covered: Dict[int, List[Tuple[float, float]]] = {}
    for idx, (a, b) in enumerate(polygon_edges(scene.polygon)):
        L = vdist(a, b)
        if L < 1.0:
            continue
        u = vunit(vsub(b, a))
        segs: List[Tuple[float, float]] = []
        for p in sol.placements:
            if obb_edge_contact_length(p.obb, a, b, tol) <= tol:
                continue
            proj = [vdot(vsub(c, a), u) for c in p.obb.corners()]
            lo = max(0.0, min(min(proj), L))
            hi = max(0.0, min(max(proj), L))
            if hi - lo > tol:
                segs.append((lo, hi))
        if segs:
            covered[idx] = segs
    used = sum(_union_len(v) for v in covered.values())
    return used, usable_wall_length(scene, sol.frame_angle)


def compute_metrics(scene: Scene, sol: Solution) -> Dict:
    room = polygon_area(scene.polygon)
    used = sum(p.item.area for p in sol.placements)
    if not sol.placements:
        used = sum(it.area for it in scene.items)      # 一件都没放下时显示"需求面积"
    reserved = sum(r.area() for r in scene.reserved)
    grid, n, m, cell = free_space_grid(scene, sol)
    free = sum(1 for v in grid if v) * cell
    biggest, fragments = _largest_free_block(grid, n, m, cell)
    wall = usable_wall_length(scene, sol.frame_angle)
    used_wall, usable_wall = wall_usage(scene, sol)
    usable = room - reserved
    return {
        "room_area": round(room, 1),
        "items_area": round(used, 1),
        "utilization": round(used / room, 4) if room > 0 else 0.0,
        "demand_ratio": round(used / usable, 4) if usable > 0 else 0.0,
        "free_area": round(free, 1),
        "free_ratio": round(free / room, 4) if room > 0 else 0.0,
        "largest_free_area": round(biggest, 1),
        "free_fragments": fragments,
        "reserved_area": round(reserved, 1),
        "usable_wall_length": round(wall, 1),
        "wall_area_ratio": round(sol.wall_area_ratio, 4),
        "wall_covered_length": round(used_wall, 1),
        "wall_usage": round(used_wall / usable_wall, 4) if usable_wall > 0 else 0.0,
        "aisle_length": round(aisle_length(scene.aisle_poly), 1),
        "aisle_width": round(scene.aisle_width, 1),
        "aisle_facing": sol.aisle_facing_count,
        "doors": len(scene.doors),
        "floating": sol.floating_count,
    }


# ---------------------------------------------------------------------------
# 收尾修正
# ---------------------------------------------------------------------------
def _mtv(a: OBB, b: OBB) -> Tuple[float, Optional[Point]]:
    """最小平移向量：把 a 推离 b 需要 (深度, 单位方向)。没重叠时深度 <= 0。"""
    axes: List[Point] = []
    for r in (a, b):
        rad = math.radians(r.angle)
        axes.append((math.cos(rad), math.sin(rad)))
        axes.append((-math.sin(rad), math.cos(rad)))
    ca, cb = a.corners(), b.corners()
    best = float("inf")
    best_ax: Optional[Point] = None
    for ax in axes:
        a0, a1 = _proj_range(ca, ax)
        b0, b1 = _proj_range(cb, ax)
        pen = min(a1, b1) - max(a0, b0)
        if pen <= 0:
            return pen, None
        if pen < best:
            best, best_ax = pen, ax
    if best_ax is None:
        return 0.0, None
    d = vsub(a.center, b.center)
    sign = 1.0 if vdot(d, best_ax) >= 0 else -1.0
    return best, vmul(best_ax, sign)


def _placement_ok(scene: Scene, sol: Solution, p: Placement, others: Sequence[Placement]) -> bool:
    if not rect_inside_polygon(p.obb, scene.polygon, cfg.INSIDE_TOL):
        return False
    for z in scene.zones:
        if obb_overlap(p.obb, z, 0.0):
            return False
    for d in scene.doors:
        if segment_intersects_rect(d.points[0], d.points[1], p.obb, 0.0):
            return False
    for q in others:
        if obb_overlap(p.obb, q.obb, 0.0):
            return False
    return True


def repair_overlaps(scene: Scene, sol: Solution, margin: float = 0.05) -> int:
    """把互相压进去一点点的物品沿最小平移方向推开（通常是亚毫米级浮点误差）。

    摆放是在旋转坐标系里做的，斜朝向时"贴墙"会有零点几毫米的误差，
    两个分别贴不同墙的物品可能互相压进一点点。这里在世界坐标下做一次收尾修正，
    保证交付的结果是**零重叠**的。
    """
    fixed = 0
    for _ in range(3):
        moved_any = False
        for i in range(len(sol.placements)):
            for j in range(len(sol.placements)):
                if i == j:
                    continue
                a, b = sol.placements[i], sol.placements[j]
                depth, direction = _mtv(a.obb, b.obb)
                if direction is None or depth <= 0:
                    continue
                move = vmul(direction, depth + margin)
                old = a.obb
                a.obb = OBB(a.obb.cx + move[0], a.obb.cy + move[1],
                            a.obb.hw, a.obb.hh, a.obb.angle)
                others = [q for k, q in enumerate(sol.placements) if k != i]
                if _placement_ok(scene, sol, a, others):
                    c = rotate_point(a.obb.center, -sol.frame_angle)
                    a.rect = Rect(c[0] - a.obb.hw, c[1] - a.obb.hh,
                                  c[0] + a.obb.hw, c[1] + a.obb.hh)
                    moved_any = True
                    fixed += 1
                else:
                    a.obb = old
        if not moved_any:
            break
    return fixed


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
        if p.facing_aisle:
            d["facing_aisle"] = True
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
        "reason": sol.reason,
        "elapsed_sec": round(sol.elapsed, 3),
        "metrics": compute_metrics(scene, sol),
        "placements": items,
    }
