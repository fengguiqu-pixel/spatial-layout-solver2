"""独立自检：在世界坐标下重新检查一遍解是否满足题目所有约束。

求解器内部是在旋转坐标系里做的判定，这里用另一套代码路径复核，避免"自己判自己"。
"""

from __future__ import annotations

from typing import List, Tuple

from . import config as cfg
from .geometry import (obb_overlap, rect_inside_polygon, rotate_point, segment_intersects_rect,
                       vdist)
from .scene import Scene
from .solver import Placement, Solution

# 严格复核时的穿透容差：10 微米，既能挡住真实重叠，又不会把"紧贴摆放"误判成重叠
STRICT_TOL = 0.01


def _opening_edge_world(p: Placement, frame_angle: float) -> Tuple[tuple, tuple]:
    """冰箱开门边在世界坐标下的两端点。"""
    r = p.rect
    if p.open_side == "top":
        a, b = (r.x0, r.y1), (r.x1, r.y1)
    elif p.open_side == "bottom":
        a, b = (r.x0, r.y0), (r.x1, r.y0)
    elif p.open_side == "left":
        a, b = (r.x0, r.y0), (r.x0, r.y1)
    else:
        a, b = (r.x1, r.y0), (r.x1, r.y1)
    return rotate_point(a, frame_angle), rotate_point(b, frame_angle)


def verify(scene: Scene, sol: Solution) -> List[str]:
    """返回问题列表，空列表 = 全部通过。"""
    issues: List[str] = []
    poly = scene.polygon

    # 1. 是否全部放下
    if len(sol.placements) != len(scene.items):
        missing = {it.name for it in scene.items} - {p.item.name for p in sol.placements}
        issues.append(f"未放下的物品: {sorted(missing)}")

    # 2. 每个物品在轮廓内
    for p in sol.placements:
        if not rect_inside_polygon(p.obb, poly, cfg.INSIDE_TOL):
            issues.append(f"{p.item.name} 超出轮廓")

    # 3. 物品之间不重叠
    for i in range(len(sol.placements)):
        for j in range(i + 1, len(sol.placements)):
            a, b = sol.placements[i], sol.placements[j]
            pen = obb_overlap(a.obb, b.obb, STRICT_TOL)
            if pen:
                issues.append(f"{a.item.name} 与 {b.item.name} 重叠")

    # 4. 不挡门 / 不占内开门扇形区
    for p in sol.placements:
        if segment_intersects_rect(scene.door[0], scene.door[1], p.obb, -STRICT_TOL):
            issues.append(f"{p.item.name} 遮挡门洞")
        for rz in scene.reserved:
            if obb_overlap(p.obb, rz, STRICT_TOL):
                issues.append(f"{p.item.name} 占用内开门 N×N 区域")

    # 5. 冰箱开门边朝室内、且没有东西贴上去
    for p in sol.placements:
        if not p.item.is_fridge or p.strip is None:
            continue
        strip_world = p.strip.as_obb().rotated(sol.frame_angle)
        if not rect_inside_polygon(strip_world, poly, 1e-6):
            issues.append(f"{p.item.name} 开门边朝向墙外，无法开门")
        for q in sol.placements:
            if q is p:
                continue
            if obb_overlap(strip_world, q.obb, STRICT_TOL):
                issues.append(f"{q.item.name} 紧贴/压住 {p.item.name} 的开门边")

    return issues


def report(scene: Scene, sol: Solution) -> str:
    lines: List[str] = []
    issues = verify(scene, sol)
    lines.append(f"[{scene.name}] feasible={sol.feasible}  "
                 f"朝向={sol.frame_angle:.2f}°  "
                 f"贴墙={sol.wall_contact_count}/{len(scene.items)}  "
                 f"悬空={sol.floating_count}  "
                 f"耗时={sol.elapsed:.2f}s")
    for p in sol.placements:
        extra = f"  开门边={p.open_side}" if p.open_side else ""
        lines.append(f"    {p.item.name:<12} center=({p.obb.cx:,.1f}, {p.obb.cy:,.1f})  "
                     f"angle={p.angle:.2f}°  贴墙面={p.wall_contacts}{extra}")
    if sol.unplaced:
        lines.append(f"    未放下: {sol.unplaced}")
    if issues:
        lines.append("    自检未通过:")
        lines.extend("      - " + s for s in issues)
    else:
        lines.append("    自检通过：全部在轮廓内 / 互不重叠 / 未挡门 / 冰箱开门侧无遮挡")
    return "\n".join(lines)
