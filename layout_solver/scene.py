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
from .circulation import aisle_length, compute_aisle
from .geometry import (OBB, Point, clean_polygon, point_in_polygon, polygon_bbox, polygon_edges,
                       rotate_point, vadd, vdist, vmul, vsub, vunit)

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
    aisle_poly: List[Point] = field(default_factory=list)
    aisle_band: List[OBB] = field(default_factory=list)
    aisle_width: float = 0.0
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
        """所有硬禁放区：内开门 N×N + 动线通道。"""
        return list(self.reserved) + list(self.aisle_band)

    @property
    def aisle_area_est(self) -> float:
        """动线带面积估算（去掉重叠后的近似：中心线长 × 宽）。"""
        return aisle_length(self.aisle_poly) * self.aisle_width


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

    scene = Scene(name=name, polygon=poly, doors=doors, items=items, reserved=reserved,
                  door_inward=inward_norms[0],
                  exit_inward=inward_norms[1] if len(inward_norms) > 1 else inward_norms[0])

    width = cfg.AISLE_WIDTH if aisle_width is None else aisle_width
    if cfg.ENABLE_AISLE and width > 0:
        build_aisle(scene, width)
    return scene


def build_aisle(scene: Scene, width: float, mode: str = "center") -> None:
    """（重）算动线，写回 scene。"""
    poly, band = compute_aisle(scene, width, mode)
    scene.aisle_poly = poly
    scene.aisle_band = band
    scene.aisle_width = width if band else 0.0


def load_scene(path: str, aisle_width: Optional[float] = None) -> Scene:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return build_scene(data, name=os.path.splitext(os.path.basename(path))[0],
                       aisle_width=aisle_width)


def polygon_bbox_of(scene: Scene) -> Tuple[float, float, float, float]:
    return polygon_bbox(scene.polygon)


def polygon_perimeter(poly: List[Point]) -> float:
    return sum(vdist(a, b) for a, b in polygon_edges(poly))
