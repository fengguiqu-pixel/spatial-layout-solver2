"""输入解析：把题目给的 JSON 变成内部 Scene 对象。

输入字段（题目给定）：
    boundary      轮廓顶点列表，首尾相连
    door          门的两端点
    isOpenInward  是否内开门
    algoToPlace   {名称: [length, width]}，名称形如 fridge / shelf-1 / overShelf-2
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config as cfg
from .geometry import (OBB, Point, clean_polygon, point_in_polygon, polygon_edges,
                       rotate_point, vadd, vdist, vmul, vsub, vunit)

# 物品类型：从名称里去掉 "-1" 之类的序号后缀
KNOWN_TYPES = ("fridge", "shelf", "overShelf", "iceMaker")


def parse_item_type(name: str) -> str:
    base = name.split("-")[0].split("#")[0].strip()
    return base if base in KNOWN_TYPES else "unknown"


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
    door: Tuple[Point, Point]
    is_open_inward: bool
    items: List[Item]
    reserved: List[OBB] = field(default_factory=list)   # 内开门占用的 N×N 空间
    door_inward: Point = (0.0, 0.0)
    door_width: float = 0.0

    @property
    def door_length(self) -> float:
        return vdist(self.door[0], self.door[1])


def _inward_normal(seg: Tuple[Point, Point], poly: List[Point], probe: float = 1.0) -> Point:
    """经验法求门所在边的"朝室内"法向：向两侧各探 probe 距离，看哪侧在轮廓内。"""
    a, b = seg
    u = vunit(vsub(b, a))
    n = (-u[1], u[0])
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    if point_in_polygon(vadd(mid, vmul(n, probe)), poly, 1e-6) == 1:
        return n
    if point_in_polygon(vsub(mid, vmul(n, probe)), poly, 1e-6) == 1:
        return (-n[0], -n[1])
    return n


def build_scene(data: Dict, name: str = "scene") -> Scene:
    poly = clean_polygon([(float(p[0]), float(p[1])) for p in data["boundary"]])
    door = ((float(data["door"][0][0]), float(data["door"][0][1])),
            (float(data["door"][1][0]), float(data["door"][1][1])))
    is_inward = bool(data.get("isOpenInward", False))

    items = []
    for key, dims in data["algoToPlace"].items():
        length, width = float(dims[0]), float(dims[1])
        items.append(Item(name=key, kind=parse_item_type(key), length=length, width=width))
    # 大件优先：约束多的先放，成功率更高
    items.sort(key=lambda it: (-it.area, it.name))

    n = _inward_normal(door, poly)
    reserved: List[OBB] = []
    if is_inward:
        N = vdist(door[0], door[1])
        mid = ((door[0][0] + door[1][0]) / 2.0, (door[0][1] + door[1][1]) / 2.0)
        center = vadd(mid, vmul(n, N / 2.0))
        import math
        ang = math.degrees(math.atan2(door[1][1] - door[0][1], door[1][0] - door[0][0]))
        reserved.append(OBB(center[0], center[1], N / 2.0, N / 2.0, ang))

    return Scene(name=name, polygon=poly, door=door, is_open_inward=is_inward,
                 items=items, reserved=reserved, door_inward=n, door_width=vdist(door[0], door[1]))


def load_scene(path: str) -> Scene:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return build_scene(data, name=os.path.splitext(os.path.basename(path))[0])


def polygon_bbox(poly: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_perimeter(poly: List[Point]) -> float:
    return sum(vdist(a, b) for a, b in polygon_edges(poly))
