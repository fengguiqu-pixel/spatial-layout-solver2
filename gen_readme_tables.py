"""临时脚本：把 main.py 的运行日志转成 README 第三节的样例表格。"""
import io, re, json, sys

log = io.open(sys.argv[1], encoding='utf-8').read().splitlines()
sizes = {}
for i in range(1, 8):
    d = json.load(io.open(f"examples/example{i}.json", encoding='utf-8'))
    sizes[i] = d["algoToPlace"]

blocks = []
cur = None
for line in log:
    m = re.match(r"\[(example\d+)\] (.*)", line)
    if m:
        if cur:
            blocks.append(cur)
        cur = {"name": m.group(1), "head": m.group(2), "rows": [], "notes": []}
        continue
    if cur is None:
        continue
    s = line.strip()
    if s.startswith("空间:") or s.startswith("通道:"):
        cur["notes"].append(s)
    elif s.startswith("说明：") or s.startswith("判定") or s.startswith("自检"):
        cur["notes"].append(s)
    elif s.startswith("未放下:"):
        cur["notes"].append(s)
    else:
        mm = re.match(r"(\S+)\s+center=\(([^)]*)\)\s+angle=([\-\d.]+)°\s+贴墙面=(\d+)(.*)", s)
        if mm:
            cur["rows"].append((mm.group(1), mm.group(2), mm.group(3), mm.group(4), mm.group(5).strip()))

if cur:
    blocks.append(cur)

TITLES = {
    1: "example1（题目给定 · 8 件 · 五边形带斜墙 · 1 扇门）",
    2: "example2（题目给定 · 8 件 · 右侧带凹口 · 1 扇门）",
    3: "example3（题目给定 · 9 件 · L 型房 · 内开门 700×700）",
    4: "example4（题目给定 · 6 件 · 带凹口的矩形 · 1 扇门）",
    5: "example5（补充 · 8 件 · example1 整体旋转 30° · 1 扇门）",
    6: "example6（补充 · 9 件 · 20° 斜墙平行四边形 · 2 扇门）",
    7: "example7（补充 · 9 件 · 小房间塞不下）",
}

out = []
for b in blocks:
    idx = int(b["name"].replace("example", ""))
    out.append(f"### {TITLES[idx]}\n")
    out.append("```")
    out.append(b["head"])
    for n in b["notes"]:
        out.append(n)
    out.append("```\n")
    if b["rows"]:
        out.append("| 物品 | 尺寸 | 中心点 | 角度 | 贴墙面数 | 备注 |")
        out.append("|---|---|---|---|---|---|")
        for name, center, ang, wc, rest in b["rows"]:
            sz = sizes[idx].get(name)
            szs = f"{int(sz[0])}×{int(sz[1])}" if sz else "–"
            rest = rest.replace("开门边=", "冰箱开门边=")
            out.append(f"| {name} | {szs} | ({center}) | {ang}° | {wc} | {rest} |")
        out.append("")
    out.append(f"![{b['name']}](output/{b['name']}.png)\n")

md = io.open("README.md", encoding='utf-8').read()
section = ("## 三、既定输入的输出示例\n\n" + "\n".join(out).rstrip() +
           "\n\n图中：灰底为房间轮廓，**绿 = 入口门 ENTER / 蓝 = 出口门 EXIT / 红 = 单门**，\n"
           "橙色虚线框为内开门 N×N 禁放区与门侧净空条，浅蓝格为**中央通道内核**（标注 KEEP CLEAR 宽度），\n"
           "深绿粗线为冰箱开门边，矩形内标注名称 / 尺寸 / 角度。\n\n---\n\n## 四、")
md2 = re.sub(r"## 三、既定输入的输出示例.*?\n---\n\n## 四、", lambda m: section, md,
             flags=re.S)
assert md2 != md or "## 三、" not in md, "第三节没匹配上"
with io.open("README.md", "w", encoding='utf-8') as f:
    f.write(md2)
print("ok")
