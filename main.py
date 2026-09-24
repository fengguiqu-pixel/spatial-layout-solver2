"""命令行入口：读取题目 JSON → 求解 → 自检 → 输出 JSON + PNG。

用法：
    python main.py                          # 跑 examples/ 下全部样例
    python main.py -i examples/example1.json
    python main.py -i examples -o output    # 整个目录
    python main.py -i xx.json --door-clearance 800   # 门洞前额外留 800mm 通道
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from layout_solver import config as cfg
from layout_solver.scene import load_scene
from layout_solver.solver import solve, solution_to_dict
from layout_solver.verify import report
from layout_solver.visualize import _HAS_PIL, render_png

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "examples")


def collect_inputs(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".json"))
    return [path]


def run_one(path: str, outdir: str, door_clearance: float, png: bool) -> dict:
    scene = load_scene(path)
    sol = solve(scene, door_clearance=door_clearance)
    print(report(scene, sol))

    os.makedirs(outdir, exist_ok=True)
    data = solution_to_dict(scene, sol)
    json_path = os.path.join(outdir, f"{scene.name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if png:
        if _HAS_PIL:
            p = render_png(scene, sol, os.path.join(outdir, f"{scene.name}.png"))
            print(f"    图片: {p}")
        else:
            print("    未安装 Pillow，跳过出图（pip install pillow）")
    print(f"    结果: {json_path}")
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="矩形物体在任意多边形轮廓内的摆放求解器")
    ap.add_argument("-i", "--input", default=DEFAULT_INPUT, help="输入 JSON 文件或目录")
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "output"), help="输出目录")
    ap.add_argument("--door-clearance", type=float, default=None,
                    help=f"门洞前额外禁放深度（默认 {cfg.DOOR_CLEARANCE_DEPTH}）")
    ap.add_argument("--no-png", action="store_true", help="不生成 PNG，只输出 JSON")
    ap.add_argument("--compact", action="store_true",
                    help="紧凑模式：物品往一起挤，让剩余空地连成整块（与默认贴墙模式对比用）")
    ap.add_argument("--grid-cells", type=int, default=None,
                    help="估算剩余空地的栅格采样格数（默认 40000，越大越准越慢）")
    args = ap.parse_args(argv)
    if args.compact:
        cfg.COMPACT_MODE = True
    if args.grid_cells:
        cfg.GRID_CELLS = args.grid_cells

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"没找到输入文件: {args.input}")
        return 1

    mode = "紧凑模式" if cfg.COMPACT_MODE else "贴墙模式"
    print(f"运行模式: {mode}   空地采样格数: {cfg.GRID_CELLS}")
    for path in inputs:
        run_one(path, args.output, args.door_clearance, not args.no_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
