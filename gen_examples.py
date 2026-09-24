import json, math, os

def rot(p, deg, o):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    x, y = p[0]-o[0], p[1]-o[1]
    return [o[0] + x*c - y*s, o[1] + x*s + y*c]

base = json.load(open('examples/example1.json', encoding='utf-8'))
pts = base['boundary'][:-1] if base['boundary'][0] == base['boundary'][-1] else base['boundary']
cx = sum(p[0] for p in pts)/len(pts); cy = sum(p[1] for p in pts)/len(pts)
o = (cx, cy)
rotated = [rot(p, 30.0, o) for p in pts]
rotated.append(rotated[0])
ex5 = {
    "boundary": [[round(x, 4), round(y, 4)] for x, y in rotated],
    "door": [[round(v, 4) for v in rot(base['door'][0], 30.0, o)],
             [round(v, 4) for v in rot(base['door'][1], 30.0, o)]],
    "isOpenInward": False,
    "algoToPlace": base['algoToPlace'],
}
json.dump(ex5, open('examples/example5.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

# example6: 20 度斜墙平行四边形房间 + 内开门
a = math.radians(20.0)
u = (math.cos(a), math.sin(a)); v = (-math.sin(a), math.cos(a))
P0 = (50000.0, 50000.0)
def add(p, d, l): return (p[0] + d[0]*l, p[1] + d[1]*l)
P1 = add(P0, u, 4200.0)
P2 = add(P1, v, 2400.0)
P3 = add(P2, u, -4200.0)
door_a = add(P0, u, 1200.0); door_b = add(P0, u, 2000.0)
# 顶边（P3 -> P2）上再开一个出口门
exit_a = add(P3, u, 1400.0); exit_b = add(P3, u, 2200.0)
poly = [P0, P1, P2, P3, P0]
r = lambda p: [round(p[0], 4), round(p[1], 4)]
ex6 = {
    "boundary": [[round(x, 4), round(y, 4)] for x, y in poly],
    "doors": [
        {"points": [r(door_a), r(door_b)], "isOpenInward": True, "role": "enter"},
        {"points": [r(exit_a), r(exit_b)], "isOpenInward": False, "role": "exit"},
    ],
    "algoToPlace": {
        "fridge": [1220, 1330], "iceMaker": [760, 850],
        "shelf-1": [1000, 400], "shelf-2": [1000, 400], "shelf-3": [1000, 400], "shelf-4": [1000, 400],
        "overShelf-1": [600, 400], "overShelf-2": [600, 400], "overShelf-3": [600, 400]
    }
}
json.dump(ex6, open('examples/example6.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

# example7: 小房间塞太多东西 -> 预期放不下
ex7 = {
    "boundary": [[30000, 20000], [32200, 20000], [32200, 22200], [30000, 22200], [30000, 20000]],
    "door": [[30000, 21400], [30000, 20800]],
    "isOpenInward": True,
    "algoToPlace": {
        "fridge": [1220, 1330],
        "shelf-1": [1000, 400], "shelf-2": [1000, 400], "shelf-3": [1000, 400], "shelf-4": [1000, 400],
        "overShelf-1": [600, 400], "overShelf-2": [600, 400], "overShelf-3": [600, 400], "overShelf-4": [600, 400]
    }
}
json.dump(ex7, open('examples/example7.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
print('generated')
