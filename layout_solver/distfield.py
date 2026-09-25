"""距离场与中央内核。

**物品全部贴墙、沿墙左右紧邻，中央自然剩下一整块**——于是"人能不能走过去"不需要
搜索路径（上一版的栅格 Dijkstra 动线就是这么被换掉的：通道带会把房间切开，切完
剩下的碎块放不进大件，又得逐级收窄、换走法，最后还可能整题无解）。

本模块只保留两样东西：

1. ``DistField``：房间内"到最近墙面的距离"栅格场，世界坐标，只算一次。给中央
   通道**画出来**（可视化 + 面积估算）用，以及作为"哪里是房间深处"的粗粒度参考。
2. ``core_distance``：把"要留多宽的通道"反解成内核阈值，只在建场时算一次。

注意：**真正的通行判据不在这里**。摆放时用的是 solver 里的 ``front_clear_ok``
（设备朝内那一面到最近障碍的净空，局部、便宜）；出结果后再用 ``corridor_report``
（自由空间栅格 + 距离变换 + 腐蚀 + 连通性 + 门可达）独立复核一遍。

别拿这个距离场去判"某条边是不是朝向房间内部"——它取的是到**最近**墙面的距离，
贴着一面长墙摆的时候整条带子都被那面墙支配，沿墙方向走多远数值都不变，判不出来。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from . import config as cfg
from .geometry import Point, point_in_polygon, point_segment_distance, polygon_bbox, polygon_edges


class DistField:
    """房间内"到最近墙面的距离"栅格场（世界坐标，只算一次）。

    外部格子记为 0 —— 物品本来就必须在轮廓内，越界由 rect_inside_polygon 负责抓。
    """

    __slots__ = ("x0", "y0", "cw", "ch", "n", "m", "grid", "maxd", "cell_area", "poly")

    def __init__(self, poly: Sequence[Point], cells: int = None):
        if cells is None:
            cells = cfg.DIST_FIELD_CELLS
        self.poly = list(poly)
        x0, y0, x1, y1 = polygon_bbox(poly)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            self.n = self.m = 1
            self.grid = [0.0]
            self.maxd = 0.0
            self.x0, self.y0, self.cw, self.ch = x0, y0, 1.0, 1.0
            self.cell_area = 1.0
            return
        n = max(8, int(round(math.sqrt(cells * w / h))))
        m = max(8, int(round(cells / n)))
        self.n, self.m = n, m
        self.cw, self.ch = w / n, h / m
        self.x0, self.y0 = x0, y0
        self.cell_area = self.cw * self.ch

        edges = polygon_edges(poly)
        grid = [0.0] * (n * m)
        maxd = 0.0
        for i in range(n):
            x = x0 + (i + 0.5) * self.cw
            for j in range(m):
                y = y0 + (j + 0.5) * self.ch
                if point_in_polygon((x, y), poly, 1e-6) != 1:
                    continue
                d = min(point_segment_distance((x, y), a, b) for a, b in edges)
                grid[i * m + j] = d
                if d > maxd:
                    maxd = d
        self.grid = grid
        self.maxd = maxd

    # ---- 查询 ----
    def dist(self, p: Point) -> float:
        i = int((p[0] - self.x0) / self.cw)
        j = int((p[1] - self.y0) / self.ch)
        if i < 0 or j < 0 or i >= self.n or j >= self.m:
            return 0.0
        return self.grid[i * self.m + j]

    def cell_center(self, i: int, j: int) -> Point:
        return (self.x0 + (i + 0.5) * self.cw, self.y0 + (j + 0.5) * self.ch)

    def core_cells(self, core: float) -> List[Tuple[int, int]]:
        """内核覆盖的格子（供可视化画通道）。"""
        out: List[Tuple[int, int]] = []
        if core <= 0:
            return out
        for i in range(self.n):
            base = i * self.m
            for j in range(self.m):
                if self.grid[base + j] >= core:
                    out.append((i, j))
        return out

    def core_area(self, core: float) -> float:
        return len(self.core_cells(core)) * self.cell_area


def core_distance(field: DistField, aisle_width: float) -> float:
    """保证中央通道宽 >= aisle_width 时，物品最内侧允许到达的 dist 上限。"""
    if not cfg.ENABLE_AISLE or aisle_width <= 0:
        return field.maxd
    core = field.maxd - aisle_width / 2.0
    return max(core, field.maxd * 0.15)
