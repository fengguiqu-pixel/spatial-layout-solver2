"""输入解析：把题目给的 JSON 变成内部 Scene 对象。

支持的输入字段
--------------
    boundary      轮廓顶点列表，首尾相连
    doors         （推荐）门列表，每项 {"points": [[x1,y1],[x2,y2]],
                                       "isOpenInward": bool, "role": "enter"/"exit"}
                  内开门会占据门宽 N 的 N×N 空间
    door / isOpenInward
                  （兼容）单门写法，等价于 doors = [{"points": door, ...}]
    algoToPlace   {名称: [length, width]}，名称形如 fridge / shelf-1 / overShelf-2

门可以是一扇也可以是两扇。两扇时动线 = 入口门 → 出口门；只有一扇时该门兼作出入口，
动线 = 门 → 房间最深处。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config as cfg
from .distfield import DistField, core_distance
from .geometry import (OBB, Point, clean_polygon, edge_angle_deg, point_in_polygon, polygon_bbox,
                       polygon_edges, rotate_point, vadd, vdist, vmul, vsub, vunit)

# 物品类型：从名称里去掉 "-1" 之类的序号后缀
KNOWN_TYPES = ("fridge", "shelf", "overShelf", "iceMaker")


def parse_item_type(name: str) -> str:
    base = name.split("-")[0].split("#")[0].strip()
    return base if base in KNOWN_TYPES else "unknown"


@dataclass
class Door:
    points: Tuple[Point, Point]
    is_open_inward: bool = False
    role: str = "both"          # enter / exit / both

    @property
    def width(self) -> float:
        return vdist(self.points[0], self.points[1])

    @property
    def mid(self) -> Point:
        return ((self.points[0][0] + self.points[1][0]) / 2.0,
                (self.points[0][1] + self.points[1][1]) / 2.0)


@dataclass
class Item:
    name: str
    kind: str
    length: float
    width: float

    @property
    def area(self) -> float:
        return self.length * self.width

    @property
    def is_fridge(self) -> bool:
        return self.kind == "fridge"


@dataclass
class Scene:
    name: str
    polygon: List[Point]
    doors: List[Door]
    items: List[Item]
    reserved: List[OBB] = field(default_factory=list)   # 内开门占用的 N×N 空间
    door_clear: List[OBB] = field(default_factory=list)  # 门洞 + 两侧净空的薄禁放条
    field: Optional[DistField] = None   # 到墙面的距离场（世界坐标，只算一次）
    core_dist: float = 0.0              # 物品最内侧允许到达的 dist 上限
    aisle_width: float = 0.0            # 实际采用的中央通道保证宽度
    door_inward: Point = (0.0, 0.0)      # 入口门的朝内法向
    exit_inward: Point = (0.0, 0.0)

    # ---- 门 ----
    @property
    def enter_door(self) -> Tuple[Point, Point]:
        for d in self.doors:
            if d.role == "enter":
                return d.points
        return self.doors[0].points

    @property
    def exit_door(self) -> Optional[Tuple[Point, Point]]:
        for d in self.doors:
            if d.role == "exit":
                return d.points
        return None

    @property
    def door_width(self) -> float:
        return self.doors[0].width

    @property
    def has_two_doors(self) -> bool:
        return self.exit_door is not None

    # ---- 禁放区 ----
    @property
    def zones(self) -> List[OBB]:
        """所有硬禁放区：内开门 N×N + 门洞两侧净空条。"""
        return list(self.reserved) + list(self.door_clear)

    @property
    def aisle_area_est(self) -> float:
        """中央内核的面积估算（quick_infeasibility 里要把它从可用面积里扣掉）。"""
        if self.field is None or self.core_dist <= 0:
            return 0.0
        return self.field.core_area(self.core_dist)


def _inward_normal(seg: Tuple[Point, Point], poly: List[Point], probe: float = 1.0) -> Point:
    """经验法求门所在边的"朝室内"法向。"""
    a, b = seg
    u = vunit(vsub(b, a))
    n = (-u[1], u[0])
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    if point_in_polygon(vadd(mid, vmul(n, probe)), poly, 1e-6) == 1:
        return n
    if point_in_polygon(vsub(mid, vmul(n, probe)), poly, 1e-6) == 1:
        return (-n[0], -n[1])
    return n


def _parse_doors(data: Dict) -> List[Door]:
    doors: List[Door] = []
    raw = data.get("doors")
    if raw:
        for d in raw:
            pts = d.get("points") or d.get("door") or d.get("segment")
            points = ((float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1])))
            doors.append(Door(points=points,
                              is_open_inward=bool(d.get("isOpenInward", False)),
                              role=str(d.get("role", "both")).lower()))
    elif "door" in data:
        d = data["door"]
        points = ((float(d[0][0]), float(d[0][1])), (float(d[1][0]), float(d[1][1])))
        doors.append(Door(points=points,
                          is_open_inward=bool(data.get("isOpenInward", False)),
                          role=str(data.get("doorRole", "both")).lower()))
    # 角色补齐：两扇门但没标 role 时，第一个当入口第二个当出口
    if len(doors) >= 2 and all(d.role == "both" for d in doors):
        doors[0].role = "enter"
        doors[1].role = "exit"
    return doors


def build_scene(data: Dict, name: str = "scene", aisle_width: Optional[float] = None) -> Scene:
    poly = clean_polygon([(float(p[0]), float(p[1])) for p in data["boundary"]])
    doors = _parse_doors(data)
    if not doors:
        raise ValueError("输入里没有门：需要 door / doors 字段")

    items = []
    for key, dims in data["algoToPlace"].items():
        length, width = float(dims[0]), float(dims[1])
        items.append(Item(name=key, kind=parse_item_type(key), length=length, width=width))
    # 大件优先：约束多的先放，成功率更高
    items.sort(key=lambda it: (-it.area, it.name))

    # 内开门的 N×N 禁放区
    reserved: List[OBB] = []
    inward_norms: List[Point] = []
    for d in doors:
        n = _inward_normal(d.points, poly)
        inward_norms.append(n)
        if d.is_open_inward:
            N = d.width
            center = vadd(d.mid, vmul(n, N / 2.0))
            ang = math.degrees(math.atan2(d.points[1][1] - d.points[0][1],
                                          d.points[1][0] - d.points[0][0]))
            reserved.append(OBB(center[0], center[1], N / 2.0, N / 2.0, ang))

    # 门洞两侧净空：沿门所在的墙面向两侧各让出 DOOR_SIDE_CLEARANCE，这段墙不许贴东西。
    # 做成一条很薄的禁放条（只有 DOOR_CLEAR_DEPTH 厚）——它的作用是"占住墙面"，
    # 不是真的要占掉一块面积。
    door_clear: List[OBB] = []
    depth = cfg.DOOR_CLEAR_DEPTH
    if depth > 0:
        for d, n in zip(doors, inward_norms):
            a, b = d.points
            half = d.width / 2.0 + cfg.DOOR_SIDE_CLEARANCE
            center = vadd(d.mid, vmul(n, depth / 2.0))
            door_clear.append(OBB(center[0], center[1], half, depth / 2.0, edge_angle_deg(a, b)))

    scene = Scene(name=name, polygon=poly, doors=doors, items=items, reserved=reserved,
                  door_clear=door_clear, field=DistField(poly),
                  door_inward=inward_norms[0],
                  exit_inward=inward_norms[1] if len(inward_norms) > 1 else inward_norms[0])

    width = cfg.AISLE_WIDTH if aisle_width is None else aisle_width
    build_core(scene, width)
    return scene


def build_core(scene: Scene, width: float) -> None:
    """（重）算中央内核：给定要保留的通道宽度，反解物品最深能摆到哪。"""
    if not cfg.ENABLE_AISLE or width <= 0 or scene.field is None:
        scene.aisle_width = 0.0
        scene.core_dist = 0.0
        return
    scene.aisle_width = width
    scene.core_dist = core_distance(scene.field, width)


def load_scene(path: str, aisle_width: Optional[float] = None) -> Scene:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return build_scene(data, name=os.path.splitext(os.path.basename(path))[0],
                       aisle_width=aisle_width)


def polygon_bbox_of(scene: Scene) -> Tuple[float, float, float, float]:
    return polygon_bbox(scene.polygon)


def polygon_perimeter(poly: List[Point]) -> float:
    return sum(vdist(a, b) for a, b in polygon_edges(poly))
