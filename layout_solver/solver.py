"""核心求解器：**沿墙环形单层排布**。

摆放规则（对应题目要求 + 门店实际规范）
--------------------------------------
1. 每件物品**贴墙**：长边与墙面共线，斜墙也一样（斜着贴）。
2. 贴不下时只能沿墙面方向**左右紧邻**另一件物品：即贴在它"沿墙切向"的两个端面上，
   排成一排继续往两边延伸。
3. **不许垂直于墙面往房间里凸出去**摆第二层——候选位置只从"真实墙面"和"已放物品的
   左右端面"生成，朝内那一面在标准层里根本不是可贴的面，结构上就不可能凸出去。
4. 全部贴墙之后，房间中央自然剩下一整块空地 = 通道。所以"人能不能走过去"不需要搜索
   路径，只要保证物品**没有侵占中央内核**（见 distfield.py）即可。
5. 沿墙排不完时按 RING_LEVELS 降级：先允许"再贴一圈"（贴第一圈朝内那一面，仍然不得
   侵占内核），再不行才允许摆到房间中央（放弃通道保证）。

关键设计
--------
1. **朝向枚举**：合法朝向由轮廓边方向决定（mod 90° 聚类），逐族求解取最优。
2. **候选生成**：把矩形贴到某个"可贴的面"上再沿面滑动，滑动点取面两端、所有顶点/已放
   物品/禁区的边界投影 + 等距采样。这样既能贴角、也能紧贴已放物品，不会悬空。
3. **打分**：贴墙级别 → 新占用的墙长（奖励去占还没人用的墙段，避免物品扎堆、墙边空着）
   → 接触长度 → 紧凑度。
4. **回溯**：贪心一次不成就按打分顺序换候选重来，受节点数与时间预算限制。
5. **特殊约束**：门洞本体不可压；门两侧各留 DOOR_SIDE_CLEARANCE 的墙面不许贴；
   内开门 N×N 直接作为禁放区；冰箱开门边必须朝室内且不许有物品相贴。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import config as cfg
from .geometry import (OBB, Point, Rect, edge_angle_deg, norm_angle_180, obb_edge_contact_length,
                       obb_overlap, point_in_polygon, polygon_area, polygon_bbox, polygon_edges,
                       rect_inside_polygon, rect_rect_contact, _proj_range,
                       obb_penetration, ray_distance_to_boundary, rotate_point,
                       segment_intersects_rect, vadd, vdist,
                       vdot, vlen, vmul, vsub, vunit)
from .scene import Item, Scene, build_core


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
    reserved: List[OBB]                       # 内开门 N×N（旋转后），可作为贴面
    edges: List[Tuple[Point, Point]] = field(default_factory=list)   # poly 的边，缓存一次
    zones_obb: List[OBB] = field(default_factory=list)   # 全部硬禁放区（含门侧净空条）
    solid_obb: List[OBB] = field(default_factory=list)   # 人**走不过去**的障碍：内开门门扇
    face_rects: List[Rect] = field(default_factory=list)  # 可作为贴面的禁放区（轴对齐的）
    field: object = None                      # DistField（世界坐标）
    core: float = 0.0                         # 物品最内侧允许到达的 dist 上限
    aisle_width: float = 0.0                  # 中央通道保证宽度
    access_relaxed: bool = False              # 通道已收窄时，操作净空也跟着降到下限
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
    facing_core: bool = False      # 交互面是否朝向房间中央（贴墙时恒为真）
    ring_level: int = 0            # 0=贴墙/左右紧邻 1=再贴一圈 2=中央
    wall_dir: Optional[str] = None  # 自己贴的那面墙的方向 'h'/'v'（决定哪两条边算"左右"）


@dataclass
class Solution:
    feasible: bool
    frame_angle: float
    placements: List[Placement] = field(default_factory=list)
    unplaced: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    nodes: int = 0
    reason: str = ""          # 判定不可行时给的原因（快速预检命中时才有）
    aisle_width: float = 0.0  # 实际采用的通道保证宽度（0 表示没留通道）
    ring_level: str = "wall"  # 实际采用的降级层级
    wall_usage: float = 0.0   # 墙边利用率：被物品占用的墙长 / 可贴墙总长

    @property
    def total_wall_contact(self) -> float:
        return sum(p.contact_len for p in self.placements)

    @property
    def wall_contact_count(self) -> int:
        return sum(1 for p in self.placements if p.wall_contacts > 0)

    @property
    def wall_area_ratio(self) -> float:
        """贴在墙上的物品面积占比：大件贴墙比小件贴墙更有意义。"""
        total = sum(p.item.area for p in self.placements)
        if total <= 0:
            return 0.0
        return sum(p.item.area for p in self.placements if p.wall_contacts > 0) / total

    @property
    def aisle_facing_count(self) -> int:
        return sum(1 for p in self.placements if p.facing_core)

    @property
    def floating_count(self) -> int:
        """既没贴墙也没贴任何已放物品的物品数量（标准层应为 0）。"""
        return sum(1 for p in self.placements if p.wall_contacts == 0 and p.item_contacts == 0)


def _rank_key(sol: "Solution") -> Tuple:
    """方案排序键（越大越好）。

    空间利用率由输入决定、不是算法能优化的东西，所以不进排序。真正体现摆放质量的
    是：贴墙的件数（越多越好）→ 墙边利用率（墙有没有被浪费）→ 贴墙面积占比。
    """
    return (sol.wall_contact_count, sol.wall_usage, sol.wall_area_ratio,
            sol.total_wall_contact)


def _pick_key(res: Sequence["Placement"]) -> Tuple:
    """同一层里两个方案谁更好：先看件数，件数相同看贴墙件数，再看贴墙总长。

    件数相同的时候必须再比贴墙——否则"这件摆到第一排外面"和"干脆不放"会被判成
    一样好，而题目要求的是尽量贴墙。
    """
    return (len(res),
            sum(1 for p in res if p.wall_contacts > 0),
            sum(p.contact_len for p in res))


def _better(a: Optional[Sequence["Placement"]],
            b: Optional[Sequence["Placement"]]) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return _pick_key(a) > _pick_key(b)


class _Budget:
    """搜索预算。**节点数是主约束**（保证结果可复现），时间只作兜底闸。

    墙钟时间不能当主约束：同一份输入在负载不同的机器上会搜出不同的解，交付出去的
    结果第二天就对不上了。
    """

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


def build_frame(scene: Scene, angle: float, extra_reserved: Sequence[OBB] = (),
                access_relaxed: bool = False) -> Frame:
    poly = [rotate_point(p, -angle) for p in scene.polygon]
    door = (rotate_point(scene.enter_door[0], -angle), rotate_point(scene.enter_door[1], -angle))
    reserved = [r.rotated(-angle) for r in scene.reserved]
    reserved.extend(r.rotated(-angle) for r in extra_reserved)
    zones = [r.rotated(-angle) for r in scene.zones]
    zones.extend(r.rotated(-angle) for r in extra_reserved)
    face_rects = [_obb_to_rect(o) for o in reserved if _is_axis_aligned(o)]

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
    # 禁放区分两种：门扇（内开门 N×N）是实心的，人走不过去；门洞前的通行缓冲和门侧
    # 净空条只是"不许摆设备"，那块地板人是可以站、可以走的。判"设备前面站得下人吗"
    # 时只能算前一种——把通行缓冲当墙，等于要求门口永远不能有人，房间会瞬间变紧。
    solid = [r.rotated(-angle) for r in scene.reserved]
    return Frame(angle=angle, poly=poly, walls=walls, door=door,
                 edges=polygon_edges(poly), reserved=reserved,
                 zones_obb=zones, solid_obb=solid, face_rects=face_rects,
                 field=scene.field, core=scene.core_dist, aisle_width=scene.aisle_width,
                 access_relaxed=access_relaxed,
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


def _rect_faces(r: Rect, wall_dir: Optional[str], level: int) -> List[Wall]:
    """已放物品有哪些边能当"可贴的面"。

    这是"只能左右紧邻、不许垂直墙面凸出"这条规则的落点：

    * 与所贴墙面**垂直**的两条边 = 该物品沿墙方向的左右端面 → 任何层级都能贴；
    * 与所贴墙面**平行**的两条边（贴墙那面 + 朝内那面）→ 贴上去就等于往房间里凸出
      一层，标准层（level 0）不允许，降级到 level >= 1 才放开。
    """
    vertical = [Wall("v", r.x0, r.y0, r.y1, -1), Wall("v", r.x1, r.y0, r.y1, +1)]
    horizontal = [Wall("h", r.y0, r.x0, r.x1, -1), Wall("h", r.y1, r.x0, r.x1, +1)]
    if wall_dir == "v":
        lateral, parallel = horizontal, vertical
    else:
        lateral, parallel = vertical, horizontal
    if level == 0:
        return lateral
    return lateral + parallel


def contact_faces(frame: Frame, placed: Sequence[Placement], forbidden: Sequence[Rect],
                  level: int) -> List[Wall]:
    """所有可贴的面：真实墙 + 已放物品的（合法）边 + 禁放区的边。"""
    faces: List[Wall] = list(frame.walls)
    for p in placed:
        lv = level
        # 第二排只能沿着第一排往外延伸，不许再贴着第二排长出第三排
        if level == 1 and p.ring_level >= 1:
            lv = 0
        faces.extend(_rect_faces(p.rect, p.wall_dir, lv))
    for r in forbidden:
        faces.extend(_rect_faces(r, None, 2))
    return faces


def gen_positions(frame: Frame, w: float, h: float, placed: Sequence[Placement],
                  forbidden: Sequence[Rect], level: int) -> List[Rect]:
    """枚举"贴在某个合法面上"的所有候选矩形（工作坐标系，轴对齐）。"""
    placed_rects = [p.rect for p in placed]
    marks_x: List[float] = [p[0] for p in frame.poly]
    marks_y: List[float] = [p[1] for p in frame.poly]
    for r in placed_rects:
        marks_x.extend((r.x0, r.x1))
        marks_y.extend((r.y0, r.y1))
    for r in forbidden:
        marks_x.extend((r.x0, r.x1))
        marks_y.extend((r.y0, r.y1))

    out: List[Rect] = []
    seen = set()
    for wall in contact_faces(frame, placed, forbidden, level):
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

    if level >= 2:
        # 摆到中央这一层不再要求贴任何面：直接按网格扫一遍，剩下的空地都能用。
        # 大件（冰箱）在只剩中央空地时特别依赖这个——贴面生成会让它们无处可去。
        bx0, by0, bx1, by1 = frame.bbox
        step = cfg.SLIDE_STEP * 2.0
        nx = max(1, int((bx1 - bx0) / step))
        ny = max(1, int((by1 - by0) / step))
        for i in range(nx + 1):
            cx = bx0 + (bx1 - bx0) * i / nx
            for j in range(ny + 1):
                cy = by0 + (by1 - by0) * j / ny
                rect = Rect(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
                key = (round(rect.x0, 2), round(rect.y0, 2))
                if key in seen:
                    continue
                seen.add(key)
                out.append(rect)
    return out


def rect_valid(frame: Frame, rect: Rect, placed: Sequence[Placement], forbidden: Sequence[Rect],
               strips: Sequence[Rect] = (), item: Optional[Item] = None,
               level: int = 0) -> bool:
    """矩形是否合法：在轮廓内、不与已放物品/禁区重叠、不压门洞、不侵占中央内核。"""
    obb = rect.as_obb()
    if not rect_inside_polygon(obb, frame.poly, cfg.INSIDE_TOL):
        return False
    for p in placed:
        if obb_overlap(obb, p.rect.as_obb(), cfg.OVERLAP_TOL):
            return False
    for r in forbidden:
        if obb_overlap(obb, r.as_obb(), cfg.OVERLAP_TOL):
            return False
    for r in strips:
        if obb_overlap(obb, r.as_obb(), cfg.STRIP_TOL):
            return False
    for z in frame.zones_obb:          # 内开门 N×N + 门两侧净空条
        if obb_overlap(obb, z, cfg.OVERLAP_TOL):
            return False
    if segment_intersects_rect(frame.door[0], frame.door[1], obb, 0.0):
        return False

    # 通道：贴墙排完之后，朝内那一面前面必须还留得下 >= 通道宽 的净空；
    # level 2（摆中央）才允许放弃这条
    if level < 2 and frame.aisle_width > 0:
        if not front_clear_ok(frame, rect, placed, item):
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
    for a, b in frame.edges:
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


def _main_wall(frame: Frame, rect: Rect) -> Optional[Wall]:
    """矩形主要贴在哪个方向的墙上。"""
    best: Optional[Tuple[float, Wall]] = None
    tol = cfg.CONTACT_TOL
    for w in frame.walls:
        if w.kind == "h":
            d = min(abs(rect.y0 - w.coord), abs(rect.y1 - w.coord))
            if d > tol:
                continue
            L = min(rect.x1, w.hi) - max(rect.x0, w.lo)
        else:
            d = min(abs(rect.x0 - w.coord), abs(rect.x1 - w.coord))
            if d > tol:
                continue
            L = min(rect.y1, w.hi) - max(rect.y0, w.lo)
        if L <= tol:
            continue
        if best is None or L > best[0]:
            best = (L, w)
    return best[1] if best else None


def _main_wall_dir(frame: Frame, rect: Rect) -> Optional[str]:
    w = _main_wall(frame, rect)
    return w.kind if w is not None else None


def _orientation_ok(frame: Frame, rect: Rect, item: Item,
                    placed: Sequence[Placement], level: int) -> bool:
    """贴墙的物品必须是"面宽沿墙、进深朝内"，即 length 与墙面共线。

    题目给的 length 是操作面（冰箱的开门面）的宽度，width 是进深：货架 1000×400
    是 1000 面宽、400 进深，冰箱 1220×1330 是 1220 面宽、1330 进深（制冰机
    760×850 同理，第一批数总比第二批小的是冰箱/制冰机，比它大的是货架）。
    把 width 沿墙、length 朝内摆，等于让设备往房间里戳出一根刺——既破坏沿墙的
    环形，也把中央通道啃掉一块。摆到中央（level 2）时没有墙可依，不再限制。
    """
    if level >= 2:
        return True
    wdir = _main_wall_dir(frame, rect)
    if wdir is None:
        # 贴在别的物品上：沿它所贴的那件物品的墙面方向延伸
        for p in placed:
            if rect_rect_contact(rect, p.rect, cfg.CONTACT_TOL) > cfg.CONTACT_TOL:
                wdir = p.wall_dir
                break
    if wdir is None:
        return True
    along = rect.h if wdir == "v" else rect.w
    return abs(along - item.length) < 1e-6


# ---------------------------------------------------------------------------
# 通道：设备朝内那一面到前方最近障碍的净空
# ---------------------------------------------------------------------------
def _ray_hit_obb(p: Point, n: Point, obb: OBB) -> Optional[float]:
    """从 p 沿单位方向 n 射出，撞到矩形的最近距离（不相交返回 None）。"""
    lp = rotate_point(p, -obb.angle, (obb.cx, obb.cy))
    lq = rotate_point((p[0] + n[0], p[1] + n[1]), -obb.angle, (obb.cx, obb.cy))
    d = (lq[0] - lp[0], lq[1] - lp[1])
    o = (lp[0] - obb.cx, lp[1] - obb.cy)
    t0, t1 = -1e18, 1e18
    for i, half in ((0, obb.hw), (1, obb.hh)):
        if abs(d[i]) < 1e-12:
            if o[i] < -half or o[i] > half:
                return None
            continue
        ta = (-half - o[i]) / d[i]
        tb = (half - o[i]) / d[i]
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return None
    if t1 < 0.0:
        return None
    return max(t0, 0.0)


def _face_points(rect: Rect, d: Point) -> List[Point]:
    """矩形在方向 d 上那一面的采样点：两端、两个四分点、中点。

    取多个点是为了"贴角"的情形——设备一端顶着侧墙时该点净空为 0，只看最小值的
    话整件设备就被误杀了；取中位数才反映"人真正站的那一段"有多宽。
    """
    if abs(d[0]) > 0.5:                       # 左右面
        x = rect.x1 if d[0] > 0 else rect.x0
        ys = [rect.y0, rect.y0 + 0.25 * rect.h, rect.center[1],
              rect.y1 - 0.25 * rect.h, rect.y1]
        return [(x, y) for y in ys]
    y = rect.y1 if d[1] > 0 else rect.y0      # 上下面
    xs = [rect.x0, rect.x0 + 0.25 * rect.w, rect.center[0],
          rect.x1 - 0.25 * rect.w, rect.x1]
    return [(x, y) for x in xs]


def _face_clearance(frame: Frame, pts: Sequence[Point], n: Point,
                    obstacles: Sequence[OBB]) -> float:
    """从 pts 各点沿单位方向 n 射出，到最近障碍（墙 / 物品 / 内开门门扇）的距离。

    取**中位数**而不是最小值：设备一端贴着侧墙、柱子或门扇区时，那个点的净空必然
    是 0，取最小值会把本来站得下人的位置误判成不行。中位数衡量的是"设备正前方
    大部分区域"的空档，和人实际站在设备中部操作的情形一致。
    """
    ds = []
    for p in pts:
        d = ray_distance_to_boundary(frame.poly, p, n, maxd=1e9)
        for o in obstacles:
            t = _ray_hit_obb(p, n, o)
            if t is not None and t < d:
                d = t
        ds.append(d)
    ds.sort()
    return ds[len(ds) // 2]


def _front_clearance(frame: Frame, rect: Rect, wall: Wall,
                     obstacles: Sequence[OBB]) -> float:
    """设备朝内那一面到前方最近障碍（墙 / 已放物品 / 禁放区）的距离。"""
    if wall.kind == "h":
        n = (0.0, float(wall.inward))
    else:
        n = (float(wall.inward), 0.0)
    return _face_clearance(frame, _face_points(rect, n), n, obstacles)


def _front_need(item: Optional[Item], relaxed: bool = False) -> float:
    """设备正前方至少要空出多少：通道宽与操作深度取大者。

    房间实在挤不出标准值时（通道已经收窄到下限），退到"人勉强站得下"的下限，
    否则大件会被永久性地判定为放不下——那不是摆放的问题，是房间的问题。
    """
    aisle = cfg.AISLE_MIN_WIDTH if relaxed else cfg.AISLE_WIDTH
    if item is None:
        return aisle
    if item.is_fridge:
        need = cfg.FRIDGE_MIN_ACCESS_DEPTH if relaxed else cfg.FRIDGE_ACCESS_DEPTH
    else:
        need = cfg.MIN_ACCESS_DEPTH if relaxed else cfg.ACCESS_DEPTH
    return max(aisle, need)


def _walk_obstacles(frame: Frame, placed: Sequence[Placement],
                    extra: Sequence[OBB] = ()) -> List[OBB]:
    """算"设备前面还有多少空档"时要躲的东西：已放的设备 + 内开门门扇（+ 额外传入的）。

    门洞前的通行缓冲、门侧净空条**不算**——它们是给人走的，人不站在那里才奇怪。
    """
    out = [p.rect.as_obb() for p in placed]
    out.extend(frame.solid_obb)
    out.extend(extra)
    return out


def front_clear_ok(frame: Frame, rect: Rect, placed: Sequence[Placement],
                   item: Optional[Item]) -> bool:
    """设备贴墙之后，它朝内的那一面前面还留得下通道吗。

    这是"人过不去"这条约束的落点：设备贴墙 → 交互面朝房间内部 → 只要它前面
    到最近的障碍（墙、另一件设备、禁放区）还有 >= 通道宽 的净空，人就走得通。
    比"对称内缩"那套判据准得多——房间窄、只有一侧贴大件时，对称判据会把它误杀。
    """
    wall = _main_wall(frame, rect)
    if wall is None:
        return True
    obstacles = _walk_obstacles(frame, placed)
    need = _front_need(item, frame.access_relaxed)
    return _front_clearance(frame, rect, wall, obstacles) >= need - 1e-6


def _score(frame: Frame, rect: Rect, placed: Sequence[Placement], forbidden: Sequence[Rect],
           cov: Optional[Dict[object, List[Tuple[float, float]]]] = None) -> Tuple:
    """候选打分（越小越好）。

    量级依次是：贴合级别 → **新占用的墙长**（越大越好）→ 接触长度 → 紧凑度。

    "新占用的墙长"是让墙边被充分利用的关键：只奖励"贴墙"的话，物品会扎堆在已放
    物品旁边，剩下的墙段白白空着；奖励去占还没人用的墙段，墙边才排得满。
    """
    obb = rect.as_obb()
    wall_contacts = 0
    wall_len = 0.0
    for a, b in frame.edges:
        L = obb_edge_contact_length(obb, a, b, cfg.CONTACT_TOL)
        if L > cfg.CONTACT_TOL:
            wall_contacts += 1
            wall_len += L
    item_contacts = 0
    item_len = 0.0
    for p in placed:
        L = rect_rect_contact(rect, p.rect, cfg.CONTACT_TOL)
        if L > cfg.CONTACT_TOL:
            item_contacts += 1
            item_len += L
    zone_contacts = sum(1 for r in forbidden
                        if rect_rect_contact(rect, r, cfg.CONTACT_TOL) > cfg.CONTACT_TOL)

    if wall_contacts >= 2:
        base = 0.0
    elif wall_contacts == 1:
        base = 1.0
    elif item_contacts >= 1:
        base = 2.0
    elif zone_contacts >= 1:
        base = 3.0
    else:
        base = 5.0

    new_wall = 0.0
    if cov is not None:
        new_wall = _cover_gain(cov, _wall_contact_segments(frame, rect))

    dist = _dist_to_cluster(rect, [p.rect for p in placed])
    return (base, -new_wall, -(wall_len + 0.5 * item_len), dist)


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
    for a, b in frame.edges:
        L = obb_edge_contact_length(obb, a, b, cfg.CONTACT_TOL)
        if L > cfg.CONTACT_TOL:
            n += 1
            total += L
    return n, total


def _item_contact_count(rect: Rect, placed: Sequence[Placement]) -> int:
    return sum(1 for p in placed
               if rect_rect_contact(rect, p.rect, cfg.CONTACT_TOL) > cfg.CONTACT_TOL)


def ranked_candidates(frame: Frame, item: Item, placed: Sequence[Placement],
                      forbidden: Sequence[Rect], strips: Sequence[Rect] = (),
                      cov: Optional[Dict[object, List[Tuple[float, float]]]] = None,
                      level: int = 0,
                      cache: Optional[Dict] = None) -> List[Tuple[Tuple, Rect]]:
    if cache is not None:
        # 回溯会把同一个局面算很多遍，而候选集只由"已放了什么"决定，缓存命中率很高
        key = (item.name, level, len(strips),
               tuple(sorted((round(p.rect.x0, 1), round(p.rect.y0, 1),
                             round(p.rect.x1, 1), round(p.rect.y1, 1)) for p in placed)))
        hit = cache.get(key)
        if hit is not None:
            return hit
    out = _ranked_candidates(frame, item, placed, forbidden, strips, cov, level)
    if cache is not None:
        cache[key] = out
    return out


def _ranked_candidates(frame: Frame, item: Item, placed: Sequence[Placement],
                       forbidden: Sequence[Rect], strips: Sequence[Rect] = (),
                       cov: Optional[Dict[object, List[Tuple[float, float]]]] = None,
                       level: int = 0) -> List[Tuple[Tuple, Rect]]:
    out: List[Tuple[Tuple, Rect]] = []
    seen = set()
    for w, h in ((item.length, item.width), (item.width, item.length)):
        for rect in gen_positions(frame, w, h, placed, forbidden, level):
            key = (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))
            if key in seen:
                continue
            seen.add(key)
            if _cheap_reject(frame, rect):
                continue
            if not _orientation_ok(frame, rect, item, placed, level):
                continue
            if not rect_valid(frame, rect, placed, forbidden, strips, item, level):
                continue
            out.append((_score(frame, rect, placed, forbidden, cov), rect))
    out.sort(key=lambda t: t[0])
    if len(out) > cfg.MAX_CANDIDATES:
        stride = len(out) / float(cfg.MAX_CANDIDATES)
        out = [out[int(i * stride)] for i in range(cfg.MAX_CANDIDATES)]
    return out


# ---------------------------------------------------------------------------
# 冰箱开门侧
# ---------------------------------------------------------------------------
def fridge_open_strip(frame: Frame, rect: Rect, item: Item, placed: Sequence[Placement],
                      forbidden: Sequence[Rect], strips: Sequence[Rect] = ()
                      ) -> Optional[Tuple[str, Rect]]:
    """挑冰箱的开门边，返回 (边名, 禁放探测带)。

    题目说"length 的其中一边为开门边"，也就是 length 是**面宽**（门/操作面那一面的
    宽度），width 是**进深**。所以冰箱贴墙时必然是后背朝墙、门开向房间内侧：开门边
    = 所贴墙面的**对面**那条边。

    上一版拿"到最近墙面的距离场"去判断哪条边朝内，那是错的：距离场取的是到**最近**
    墙面的距离，贴着一面长墙摆的时候整条带子都被那面墙支配，沿墙方向走多远距离都
    不变，于是两条候选边一起被判成"不朝内"，冰箱永远放不下。这里改用几何判据。
    """
    buf = cfg.FRIDGE_OPEN_BUFFER
    eps = 1e-6
    length_along_x = abs(rect.w - item.length) < eps
    if length_along_x:                      # length 沿 x → 上下两条边是 length 边
        options = [("bottom", Rect(rect.x0, rect.y0 - buf, rect.x1, rect.y0), (0.0, -1.0)),
                   ("top",    Rect(rect.x0, rect.y1, rect.x1, rect.y1 + buf), (0.0, 1.0))]
    else:                                   # length 沿 y → 左右两条边是 length 边
        options = [("left",  Rect(rect.x0 - buf, rect.y0, rect.x0, rect.y1), (-1.0, 0.0)),
                   ("right", Rect(rect.x1, rect.y0, rect.x1 + buf, rect.y1), (1.0, 0.0))]

    wall = _main_wall(frame, rect)
    if wall is not None:
        # 贴墙：开门边 = 所贴墙面的对面。这同时要求 length 沿着墙走（后背贴墙），
        # 否则开门边就成了垂直于墙的那条边——等于冰箱侧着贴墙、门朝墙那边开。
        if wall.kind == "h":
            want, ok = ("top" if wall.inward > 0 else "bottom"), length_along_x
        else:
            want, ok = ("right" if wall.inward > 0 else "left"), (not length_along_x)
        if not ok:
            return None
        options = [o for o in options if o[0] == want]

    need = cfg.FRIDGE_DOOR_CLEAR_MIN if frame.access_relaxed else cfg.FRIDGE_DOOR_CLEAR
    obstacles = _walk_obstacles(frame, placed)
    obstacles.extend(r.as_obb() for r in forbidden)
    obstacles.extend(r.as_obb() for r in strips)

    best: Optional[Tuple[float, str, Rect]] = None
    for side, strip, d in options:
        obb = strip.as_obb()
        if not rect_inside_polygon(obb, frame.poly, 1e-6):
            continue                          # 这一侧顶着墙，门打不开
        if any(obb_overlap(obb, p.rect.as_obb(), 0.0) for p in placed):
            continue
        if any(obb_overlap(obb, r.as_obb(), 0.0) for r in forbidden):
            continue
        if any(obb_overlap(obb, r.as_obb(), 0.0) for r in strips):
            continue
        if any(obb_overlap(obb, z, 0.0) for z in frame.zones_obb):
            continue
        if segment_intersects_rect(frame.door[0], frame.door[1], obb, 0.0):
            continue
        if _face_clearance(frame, _face_points(rect, d), d, obstacles) < need - 1e-6:
            continue                          # 门前站不下人
        clear = _face_clearance(frame, _face_points(rect, d), d, obstacles)
        if best is None or clear > best[0]:
            best = (clear, side, strip)
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


def _facing_core(frame: Frame, rect: Rect) -> bool:
    """物品是不是在环形带里（在带里，它的交互面就必然朝向中央通道）。"""
    if frame.core <= 0:
        return True
    world = rect.as_obb().rotated(frame.angle)
    return frame.field.dist((world.cx, world.cy)) < frame.core


def _dfs(frame: Frame, order: Sequence[Item], idx: int, placed: List[Placement],
         zones: List[Rect], strips: List[Rect], budget: _Budget, level: int,
         cov: Optional[Dict[object, List[Tuple[float, float]]]] = None,
         cache: Optional[Dict] = None) -> Optional[List[Placement]]:
    """深度优先摆放。放不下全部时**允许跳过**某件物品，先把放得下的都放上。

    这是"贴墙排不完的再放中间"的落点：贴墙层不追求一次放完，能贴几件贴几件，
    剩下的交给下一层降级去处理，而不是因为最后一件塞不进去就把整层判失败。

    每个分支都做两件事再比较：试着把当前物品放下去（在 MAX_BRANCH 个候选里挑，
    放完剩下的能全放上就收工，否则继续换候选看谁最终放得多），以及干脆跳过当前
    物品。哪个最终件数多就取哪个。

    注意"放完剩下的能全放上"是**唯一**的提前返回条件。上一版写的是
    ``len(res) >= len(placed)``，那对 res ⊇ placed 来说恒真，于是第一个能放的
    候选就直接定死、再也不回溯——DFS 退化成纯贪心，大件一旦占了位置后面的小件
    放不下也不回头换。
    """
    if idx >= len(order):
        return list(placed)
    if not budget.tick():
        return list(placed)
    if cov is None:
        cov = {}

    item = order[idx]
    cands = ranked_candidates(frame, item, placed, zones, strips, cov, level, cache)
    if level == 0:
        # 贴墙层：这一层的每一件都必须真的贴到墙上。"左右紧邻另一个物体"是贴在同一面
        # 墙上的延伸，不是悬在别的物体旁边——所以这里只放行 base<=1（贴墙）的候选。
        cands = [c for c in cands if c[0][0] <= 1.0]
    elif level == 1:
        cands = [c for c in cands if c[0][0] < 5.0]

    before = len(placed)
    remaining_after = len(order) - idx - 1
    best_here: Optional[List[Placement]] = None

    # 换候选的次数随深度递减：浅层（先放的大件）值得多试几个位置，深层（填满缝的
    # 小件）基本一次成功，再展开下去只是把搜索树撑爆而换不来更多件数
    tries = max(1, cfg.MAX_BRANCH - 2 * idx)
    for _, rect in cands[: tries]:
        strip = None
        if item.is_fridge:
            got = fridge_open_strip(frame, rect, item, placed, zones, strips)
            if got is None:
                continue
            side, strip = got
        else:
            side = None

        wc, clen = _wall_contact_info(frame, rect)
        ic = _item_contact_count(rect, placed)
        wdir = _main_wall_dir(frame, rect)
        if wdir is None:
            # 没贴到墙 → 继承被贴物品的墙面方向，保证"左右紧邻"是沿同一面墙延伸
            for p in placed:
                if rect_rect_contact(rect, p.rect, cfg.CONTACT_TOL) > cfg.CONTACT_TOL:
                    wdir = p.wall_dir
                    break
        pl = Placement(item=item, rect=rect, obb=_to_world(frame, rect),
                       angle=_output_angle(frame, rect, item), wall_contacts=wc,
                       contact_len=clen, item_contacts=ic, open_side=side, strip=strip,
                       facing_core=_facing_core(frame, rect), ring_level=level, wall_dir=wdir)
        placed.append(pl)
        if strip is not None:
            strips.append(strip)
        added = _wall_contact_segments(frame, rect)
        for key, lo, hi in added:
            cov.setdefault(key, []).append((lo, hi))
        res = _dfs(frame, order, idx + 1, placed, zones, strips, budget, level, cov, cache)
        if _better(res, best_here):
            best_here = res
        for key, lo, hi in added:
            try:
                cov[key].remove((lo, hi))
            except (KeyError, ValueError):
                pass
        if strip is not None:
            strips.pop()
        placed.pop()
        if best_here is not None and len(best_here) - before > remaining_after:
            break                     # 剩下的全放上了，不必再换候选
        if budget.exhausted:
            break

    if best_here is not None and len(best_here) - before > remaining_after:
        return best_here              # 完整解：跳过分支不可能更好

    # 这一件在当前层放不下（或放下反而害得别人放不下）→ 跳过它，后面的还能继续
    skip = _dfs(frame, order, idx + 1, placed, zones, strips, budget, level, cov, cache)
    if _better(skip, best_here):
        return skip
    return best_here


def _solve_level(frame: Frame, level: int, items: Sequence[Item],
                 pre_placed: Sequence[Placement], scale: float, orders: int
                 ) -> Tuple[List[Placement], int]:
    """固定朝向 + 固定降级层级下，把 items 尽量多地摆上去，返回 (摆放结果, 节点数)。

    放不满不算失败——放得下的先摆好，剩下的由调用方交给下一层降级处理。
    """
    all_orders = [
        sorted(items, key=lambda it: (-it.area, it.name)),
        sorted(items, key=lambda it: (-max(it.length, it.width), -it.area, it.name)),
        sorted(items, key=lambda it: (it.area, it.name)),
    ]
    best: Optional[List[Placement]] = None
    nodes = 0
    for order in all_orders[: max(1, orders)]:
        # scale 同时缩放节点预算：探测阶段只问"这一档通道放不放得下"，不必搜到底
        budget = _Budget(max(50, int(cfg.NODE_BUDGET * scale)), cfg.TIME_BUDGET)
        res = _dfs(frame, order, 0, list(pre_placed), reserved_rects(frame), [],
                   budget, level, None, {})
        nodes += budget.nodes
        if res is None:
            continue
        if _better(res, best):
            best = res
    return (best if best is not None else list(pre_placed)), nodes


def quick_infeasibility(scene: Scene) -> str:
    """秒级的不可行预检，避免在注定放不下的输入上白白烧搜索预算。"""
    room = polygon_area(scene.polygon)
    reserved = sum(r.area() for r in scene.zones)
    need = sum(it.area for it in scene.items)
    usable = room - reserved
    # 通道不再额外占面积——它就是"设备前方留出来的空档"，和设备本身不重叠
    if need > usable + 1e-6:
        return (f"面积不够：物品占地合计 {need:,.0f} > 可用面积 {usable:,.0f} "
                f"(房间 {room:,.0f} - 禁放区 {reserved:,.0f})")
    x0, y0, x1, y1 = polygon_bbox(scene.polygon)
    bw, bh = x1 - x0, y1 - y0
    for it in scene.items:
        if min(it.length, it.width) > min(bw, bh) + 1e-6:
            return (f"{it.name} 的最小边 {min(it.length, it.width):,.0f} "
                    f"超过房间最窄处 {min(bw, bh):,.0f}")
    return ""


def _fill_frame(scene: Scene, angle: float, levels: Sequence[int],
                door_clearance: Optional[float], t0: float,
                scale: float = 1.0, orders: int = 3,
                access_relaxed: bool = False) -> Solution:
    """在固定朝向下分级填充：贴墙层放多少算多少，剩下的交给下一层降级。

    朝向必须固定——不同朝向的坐标系不一样，混着放没法做重叠检测。
    """
    extra: List[OBB] = []
    depth = cfg.DOOR_CLEARANCE_DEPTH if door_clearance is None else door_clearance
    if depth > 0:
        a, b = scene.enter_door
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        center = vadd(mid, vmul(scene.door_inward, depth / 2.0))
        extra.append(OBB(center[0], center[1], vdist(a, b) / 2.0, depth / 2.0,
                         edge_angle_deg(a, b)))
    frame = build_frame(scene, angle, extra, access_relaxed)

    placed: List[Placement] = []
    nodes = 0
    for level in levels:
        remaining = [it for it in scene.items if all(p.item.name != it.name for p in placed)]
        if not remaining:
            break
        placed, n = _solve_level(frame, level, remaining, placed, scale, orders)
        nodes += n
        if len(placed) >= len(scene.items):
            break

    placed_names = {p.item.name for p in placed}
    unplaced = [it.name for it in scene.items if it.name not in placed_names]
    deepest_level = max((p.ring_level for p in placed), default=0)
    sol = Solution(feasible=not unplaced, frame_angle=angle, placements=placed,
                   unplaced=unplaced, elapsed=time.time() - t0, nodes=nodes)
    used, usable = wall_usage(scene, sol)
    sol.wall_usage = used / usable if usable > 0 else 0.0
    sol.aisle_width = scene.aisle_width
    sol.ring_level = (cfg.RING_LEVELS[deepest_level]
                      if deepest_level < len(cfg.RING_LEVELS) else "free")
    return sol


def _fill(scene: Scene, levels: Sequence[int], door_clearance: Optional[float], t0: float,
          scale: float = 1.0, orders: int = 3,
          access_relaxed: bool = False) -> Solution:
    """遍历所有合法朝向，取摆放效果最好的那个。"""
    frames = candidate_frames(scene)
    best: Optional[Solution] = None
    for ang in frames:
        sol = _fill_frame(scene, ang, levels, door_clearance, t0, scale, orders, access_relaxed)
        key = (len(sol.placements), _rank_key(sol))
        if best is None or key > (len(best.placements), _rank_key(best)):
            best = sol
    if best is None:
        return Solution(feasible=False, frame_angle=frames[0] if frames else 0.0,
                        unplaced=[it.name for it in scene.items],
                        elapsed=time.time() - t0, reason="没有可用的摆放朝向")
    return best


def solve(scene: Scene, door_clearance: Optional[float] = None,
          aisle_width: Optional[float] = None) -> Solution:
    """求解一个场景。返回 Solution（feasible=False 表示放不下）。

    流程：做"贴墙 + 沿墙左右紧邻"的环形单层排布；放不下就收窄通道再试；通道收到下限
    还放不下，才逐级降级——先是"贴着第一排再放一排"，最后才把剩下的摆到房间中央。
    """
    t0 = time.time()
    reason0 = quick_infeasibility(scene)
    if reason0:
        return Solution(feasible=False, frame_angle=0.0,
                        unplaced=[it.name for it in scene.items],
                        elapsed=time.time() - t0, reason=reason0)

    if not cfg.ENABLE_AISLE:
        widths: List[float] = [0.0]
    elif aisle_width is not None:
        widths = [aisle_width]
    else:
        w = cfg.AISLE_WIDTH
        widths = [w, round(w * 0.8), cfg.AISLE_MIN_WIDTH]

    # 探测：逐级收窄通道，看哪一档能全放下（保持通道 = 只用 wall / inner 两层）
    chosen: Optional[float] = None
    last: Optional[Solution] = None
    for i, w in enumerate(widths):
        build_core(scene, w)
        # 通道已经收窄时，操作净空也跟着降到"人勉强站得下"的下限
        relaxed = i > 0
        # 探测阶段每种顺序各跑一遍：只跑"大件优先"那一种是不够的——它有时会因为
        # 大件占了地方而少放几件小件，换一种顺序反而能全放下
        sol = _fill(scene, (0, 1), door_clearance, t0, scale=0.3, orders=2,
                    access_relaxed=relaxed)
        if sol.feasible:
            # 摆放时的判据是局部的（每件设备前方净空），便宜但不保证整体连通；
            # 这里用 corridor_report 独立复核一遍，门走不到就说明这一档的通道是假的
            if w <= 0 or corridor_report(scene, sol, w)["ok"]:
                chosen = w
                break
        if last is None or len(sol.placements) > len(last.placements):
            last = sol
    # 通道收到下限还放不下 → 放弃通道，最后一层可以把剩下的摆到房间中央
    if chosen is None:
        build_core(scene, 0.0)
        sol = _fill(scene, (0, 1, 2), door_clearance, t0, scale=0.3, orders=1,
                    access_relaxed=True)
        if sol.feasible:
            chosen = 0.0
        elif sol is not None:
            last = sol

    if chosen is not None:
        build_core(scene, chosen)
        levels = (0, 1, 2) if chosen <= 0 else (0, 1)
        sol = _fill(scene, levels, door_clearance, t0, scale=1.0, orders=3,
                    access_relaxed=chosen <= 0 or chosen < cfg.AISLE_WIDTH)
        repair_overlaps(scene, sol)
        if chosen > 0 and chosen < cfg.AISLE_WIDTH:
            sol.reason = (f"房间尺寸所限，{cfg.AISLE_WIDTH:.0f} 宽的通道放不下全部物品，"
                          f"已自动收窄到 {chosen:.0f}（这是本房间能容纳的最宽通道）")
        elif chosen == 0 and cfg.ENABLE_AISLE:
            sol.reason = f"空间不足以保留 {cfg.AISLE_WIDTH:.0f} 宽的通道，已取消通道约束"
        if sol.ring_level == "inner":
            sol.reason = ((sol.reason + "；") if sol.reason else "") + \
                "沿墙一排排不完，部分物品贴着第一排又放了一排"
        elif sol.ring_level == "free":
            sol.reason = ((sol.reason + "；") if sol.reason else "") + \
                "沿墙排不完，剩余物品摆到了房间中央"
        return sol

    if last is not None:
        last.reason = last.reason or "沿墙排布与中央摆放都放不下全部物品"
        return last
    return Solution(feasible=False, elapsed=time.time() - t0)


# ---------------------------------------------------------------------------
# 指标
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


def free_space_grid(scene: Scene, sol: Solution, target_cells: int = None,
                    include_zones: bool = True):
    """把地面打成栅格，标出哪些格子是空的。

    返回 (grid, n, m, cell_area)，grid 为一维 list，True 表示"在轮廓内且没被占用"。
    include_zones=False 时只把物品和内开门当障碍——复核通道要用这个口径，
    因为门洞缓冲、门侧净空那些"禁放区"本来就是留给人走的。
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

    blocked = list(scene.zones) if include_zones else list(scene.reserved)
    blocks = [p.obb for p in sol.placements] + blocked
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


# ---------------------------------------------------------------------------
# 通道复核（独立于摆放时的判据，用栅格腐蚀重算一遍）
# ---------------------------------------------------------------------------
def _inward_of(seg: Tuple[Point, Point], poly: Sequence[Point]) -> Point:
    a, b = seg
    u = vunit(vsub(b, a))
    n = (-u[1], u[0])
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    if point_in_polygon(vadd(mid, vmul(n, 1.0)), poly, 1e-6) == 1:
        return n
    return (-n[0], -n[1])


def corridor_report(scene: Scene, sol: Solution, width: Optional[float] = None,
                    cells: int = 12000) -> Dict:
    """独立复核中央通道：把自由空间腐蚀 width/2，看剩下的能不能连成一整块、门走不走得到。

    摆放时用的是"每件设备前方净空"这种局部判据，便宜但不保证整体；这里用完全不同的
    路径（栅格 + 距离变换 + 连通域）重算一遍，两边都对才算真的留出了通道。
    """
    w = scene.aisle_width if width is None else width
    if w <= 0 or not scene.polygon:
        return {"ok": True, "width": 0.0, "area": 0.0, "doors": 0, "doors_reachable": 0,
                "reason": "未要求保留通道"}

    grid, n, m, cell = free_space_grid(scene, sol, cells, include_zones=False)
    if n == 0:
        return {"ok": False, "width": w, "area": 0.0, "doors": 0, "doors_reachable": 0,
                "reason": "栅格化失败"}

    # chamfer 距离变换：每格到最近障碍（墙 / 物品 / 禁放区）的距离
    INF = 1e18
    d = [INF if v else 0.0 for v in grid]     # 空地待求距离，障碍格距离为 0
    d1, d2 = cell, cell * 1.41421356
    for i in range(n):
        base = i * m
        for j in range(m):
            k = base + j
            if d[k] == 0.0:
                continue
            b = d[k]
            if i > 0:
                b = min(b, d[k - m] + d1)
            if j > 0:
                b = min(b, d[k - 1] + d1)
            if i > 0 and j > 0:
                b = min(b, d[k - m - 1] + d2)
            if i > 0 and j < m - 1:
                b = min(b, d[k - m + 1] + d2)
            d[k] = b
    for i in range(n - 1, -1, -1):
        base = i * m
        for j in range(m - 1, -1, -1):
            k = base + j
            if d[k] == 0.0:
                continue
            b = d[k]
            if i < n - 1:
                b = min(b, d[k + m] + d1)
            if j < m - 1:
                b = min(b, d[k + 1] + d1)
            if i < n - 1 and j < m - 1:
                b = min(b, d[k + m + 1] + d2)
            if i < n - 1 and j > 0:
                b = min(b, d[k + m - 1] + d2)
            d[k] = b

    r = w / 2.0
    core = [v >= r for v in d]

    # 连通域标记
    labels = [-1] * (n * m)
    sizes: List[int] = []
    stack: List[int] = []
    for s in range(n * m):
        if not core[s] or labels[s] >= 0:
            continue
        lid = len(sizes)
        cnt = 0
        stack.append(s)
        labels[s] = lid
        while stack:
            idx = stack.pop()
            cnt += 1
            i, j = divmod(idx, m)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    nid = ni * m + nj
                    if core[nid] and labels[nid] < 0:
                        labels[nid] = lid
                        stack.append(nid)
        sizes.append(cnt)

    if not sizes:
        return {"ok": False, "width": w, "area": 0.0, "doors": len(scene.doors),
                "doors_reachable": 0, "reason": "腐蚀后没有留下任何通道"}
    main = sizes.index(max(sizes))

    x0, y0, x1, y1 = polygon_bbox(scene.polygon)
    cw, ch = (x1 - x0) / n, (y1 - y0) / m
    reach = 0
    for door in scene.doors:
        nv = _inward_of(door.points, scene.polygon)
        p = vadd(door.mid, vmul(nv, r))
        i = min(n - 1, max(0, int((p[0] - x0) / cw)))
        j = min(m - 1, max(0, int((p[1] - y0) / ch)))
        if labels[i * m + j] == main:
            reach += 1
        else:
            # 门口正好被设备挡住时，退一步看它旁边够不够宽
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < m and labels[ni * m + nj] == main:
                        reach += 1
                        break
                else:
                    continue
                break
    area = sizes[main] * cell
    ok = reach >= len(scene.doors) and area >= w * w
    reason = "" if ok else (
        f"腐蚀 {r:.0f} 后主通道 {area:,.0f}，门可达 {reach}/{len(scene.doors)}")
    return {"ok": ok, "width": w, "area": area, "doors": len(scene.doors),
            "doors_reachable": reach, "reason": reason}


def compute_metrics(scene: Scene, sol: Solution) -> Dict:
    room = polygon_area(scene.polygon)
    used = sum(p.item.area for p in sol.placements)
    if not sol.placements:
        used = sum(it.area for it in scene.items)      # 一件都没放下时显示"需求面积"
    reserved = sum(r.area() for r in scene.zones)
    grid, n, m, cell = free_space_grid(scene, sol)
    free = sum(1 for v in grid if v) * cell
    biggest, fragments = _largest_free_block(grid, n, m, cell)
    used_wall, usable_wall = wall_usage(scene, sol)
    usable = room - reserved
    core_area = scene.field.core_area(scene.core_dist) if scene.field else 0.0
    return {
        "room_area": round(room, 1),
        "items_area": round(used, 1),
        "utilization": round(used / room, 4) if room > 0 else 0.0,
        "demand_ratio": round(used / usable, 4) if usable > 0 else 0.0,
        "free_area": round(free, 1),
        "largest_free_area": round(biggest, 1),
        "free_fragments": fragments,
        "reserved_area": round(reserved, 1),
        "usable_wall_length": round(usable_wall, 1),
        "wall_covered_length": round(used_wall, 1),
        "wall_usage": round(used_wall / usable_wall, 4) if usable_wall > 0 else 0.0,
        "wall_area_ratio": round(sol.wall_area_ratio, 4),
        "aisle_width": round(scene.aisle_width, 1),
        "core_dist": round(scene.core_dist, 1),
        "core_area": round(core_area, 1),
        "facing_core": sol.aisle_facing_count,
        "ring_level": sol.ring_level,
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


def _placement_ok(scene: Scene, p: Placement, others: Sequence[Placement]) -> bool:
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
                if _placement_ok(scene, a, others):
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
        d["ring_level"] = cfg.RING_LEVELS[p.ring_level] if p.ring_level < len(cfg.RING_LEVELS) else "free"
        items.append(d)
    return {
        "feasible": sol.feasible,
        "frame_angle": round(sol.frame_angle, 3),
        "ring_level": sol.ring_level,
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
