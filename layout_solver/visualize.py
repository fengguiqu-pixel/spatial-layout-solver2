"""可视化：用 Pillow 把轮廓 / 门 / 禁放区 / 摆放结果画成 PNG。

只依赖 Pillow，且它是可选的（main.py 在 import 失败时会自动跳过出图）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .scene import Scene
from .solver import Solution, compute_metrics
from .verify import _opening_edge_world

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:      # pragma: no cover - 环境没装 Pillow 时降级
    _HAS_PIL = False

COLORS: Dict[str, Tuple[int, int, int]] = {
    "fridge": (214, 234, 248),
    "shelf": (230, 243, 226),
    "overShelf": (250, 240, 220),
    "iceMaker": (240, 226, 240),
    "unknown": (235, 235, 235),
}
OUTLINE: Dict[str, Tuple[int, int, int]] = {
    "fridge": (33, 97, 140),
    "shelf": (56, 122, 61),
    "overShelf": (160, 116, 40),
    "iceMaker": (120, 70, 130),
    "unknown": (90, 90, 90),
}


def _font(size: int):
    if not _HAS_PIL:
        return None
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_png(scene: Scene, sol: Solution, path: str,
               width: int = 1200, height: int = 900) -> Optional[str]:
    if not _HAS_PIL:
        return None

    pad = 48
    legend_h = 74
    xs = [p[0] for p in scene.polygon]
    ys = [p[1] for p in scene.polygon]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    dx, dy = maxx - minx, maxy - miny
    scale = min((width - 2 * pad) / max(dx, 1e-6), (height - 2 * pad - legend_h) / max(dy, 1e-6))
    ox = pad + ((width - 2 * pad) - dx * scale) / 2.0
    oy = pad + ((height - 2 * pad - legend_h) - dy * scale) / 2.0

    def tx(p):
        return (ox + (p[0] - minx) * scale, oy + (maxy - p[1]) * scale)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_small = _font(13)
    f_big = _font(17)

    # 房间
    poly_px = [tx(p) for p in scene.polygon]
    d.polygon(poly_px, fill=(247, 247, 247), outline=(60, 60, 60))
    for a, b in zip(poly_px, poly_px[1:] + poly_px[:1]):
        d.line([a, b], fill=(40, 40, 40), width=3)

    # 内开门 N×N 禁放区
    for rz in scene.reserved:
        pts = [tx(c) for c in rz.corners()]
        d.polygon(pts, fill=(255, 236, 214), outline=(214, 138, 51))
        for a, b in zip(pts, pts[1:] + pts[:1]):
            _dashed_line(d, a, b, (214, 138, 51), width=2, dash=7, gap=5)
        cx, cy = tx((rz.cx, rz.cy))
        d.text((cx, cy), "door swing N x N", fill=(170, 100, 20), font=f_small, anchor="mm")

    # 门
    da, db = tx(scene.door[0]), tx(scene.door[1])
    d.line([da, db], fill=(214, 40, 40), width=7)
    mid = ((da[0] + db[0]) / 2, (da[1] + db[1]) / 2)
    d.text((mid[0], mid[1] - 14), "DOOR", fill=(214, 40, 40), font=f_big, anchor="mm")

    # 物品
    for p in sol.placements:
        pts = [tx(c) for c in p.obb.corners()]
        d.polygon(pts, fill=COLORS.get(p.item.kind, COLORS["unknown"]),
                  outline=OUTLINE.get(p.item.kind, OUTLINE["unknown"]))
        for a, b in zip(pts, pts[1:] + pts[:1]):
            d.line([a, b], fill=OUTLINE.get(p.item.kind, OUTLINE["unknown"]), width=2)
        cx, cy = tx((p.obb.cx, p.obb.cy))
        d.text((cx, cy - 7), p.item.name, fill=(25, 25, 25), font=f_small, anchor="mm")
        d.text((cx, cy + 8), f"{p.item.length:.0f}x{p.item.width:.0f} @ {p.angle:.1f}deg",
               fill=(90, 90, 90), font=f_small, anchor="mm")

    # 冰箱开门边
    for p in sol.placements:
        if p.item.is_fridge and p.open_side:
            a, b = _opening_edge_world(p, sol.frame_angle)
            d.line([tx(a), tx(b)], fill=(20, 140, 90), width=5)
            mx = (tx(a)[0] + tx(b)[0]) / 2
            my = (tx(a)[1] + tx(b)[1]) / 2
            d.text((mx, my), "  opening side", fill=(20, 140, 90), font=f_small, anchor="lm")

    # 标题
    m = compute_metrics(scene, sol)
    title = (f"{scene.name}   feasible={sol.feasible}   "
             f"frame={sol.frame_angle:.2f}deg   "
             f"wall-hugging={sol.wall_contact_count}/{len(scene.items)}   "
             f"utilization={m['utilization'] * 100:.1f}%   "
             f"free={m['free_area']:,.0f} (largest block {m['largest_free_area']:,.0f})")
    d.text((pad, 14), title, fill=(20, 20, 20), font=f_big)

    # 图例
    lx, ly = pad, height - legend_h + 8
    d.text((lx, ly), "legend:", fill=(90, 90, 90), font=f_small)
    cur = lx + 52
    for kind in ("fridge", "shelf", "overShelf", "iceMaker"):
        if not any(pl.item.kind == kind for pl in sol.placements):
            continue
        d.rectangle([cur, ly + 1, cur + 15, ly + 15], fill=COLORS[kind], outline=OUTLINE[kind])
        d.text((cur + 20, ly + 1), kind, fill=(40, 40, 40), font=f_small)
        cur += 22 + 8 * len(kind) + 16
    d.text((pad, ly + 26),
           "red = door (kept clear)   orange = inward door swing area   green = fridge opening side",
           fill=(90, 90, 90), font=f_small)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
    return path


def _dashed_line(draw, a, b, color, width: int = 2, dash: int = 8, gap: int = 6) -> None:
    import math
    x0, y0 = a
    x1, y1 = b
    L = math.hypot(x1 - x0, y1 - y0)
    if L <= 1e-6:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    t = 0.0
    while t < L:
        s = min(t + dash, L)
        draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * s, y0 + uy * s)], fill=color, width=width)
        t = s + gap
