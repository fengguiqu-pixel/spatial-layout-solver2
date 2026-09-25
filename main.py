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
from typing import List, Optional

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


def run_one(path: str, outdir: str, door_clearance: float, png: bool,
            aisle_width: Optional[float] = None) -> dict:
    scene = load_scene(path)
    sol = solve(scene, door_clearance=door_clearance, aisle_width=aisle_width)
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
    ap.add_argument("--grid-cells", type=int, default=None,
                    help="估算剩余空地的栅格采样格数（只出指标，不参与决策）")
    ap.add_argument("--aisle-width", type=float, default=None,
                    help=f"中央通道保证宽度（默认 {cfg.AISLE_WIDTH:.0f}，放不下会自动收窄）")
    ap.add_argument("--no-aisle", action="store_true", help="不保留中央通道，只做贴墙摆放")
    args = ap.parse_args(argv)
    if args.grid_cells:
        cfg.GRID_CELLS = args.grid_cells
    if args.no_aisle:
        cfg.ENABLE_AISLE = False

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"没找到输入文件: {args.input}")
        return 1

    keep = f"中央通道 {cfg.AISLE_WIDTH:.0f}" if cfg.ENABLE_AISLE else "不保留通道"
    print(f"摆放规则: 贴墙 + 沿墙左右紧邻   {keep}   空地采样格数: {cfg.GRID_CELLS}")
    for path in inputs:
        run_one(path, args.output, args.door_clearance, not args.no_png, args.aisle_width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
