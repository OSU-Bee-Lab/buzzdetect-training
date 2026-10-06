"""Step-budget curves (WSD) as a standalone SVG, two panels.

Left: headline (sensitivity_exclquiet @ fpr 0.005) against training steps, one line per WSD trunk (its decayed
branches). Filled markers are budgets the --wsd-stop rule would run; the ringed one is where it ends the trunk
(the first branch whose hit@K gain on its predecessor is under the tolerance, main.py's `plateaued`); hollow
markers lie past that stop (run without the rule, e.g. chain_wsd.sh step 3).

Right: the speed / sensitivity frontier. Grey: every rung-B seed-1 cosine run at its fixed 7k budget, and its
non-dominated line (as frontier_svg.py). Coloured: each WSD trunk's branch at the rule's stop, with its other
budgets as small hollow markers on the same vertical (speed does not depend on steps). Black line: the frontier
with the stop-rule branches added.

    python tools/human/wsd_svg.py [out.svg] [--tol 1.85]     (train env; reads the comparable ladder rows and each run's curve.json)
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '05_distill'))
import dpaths as D  # noqa: E402
import ladder_record as LR  # noqa: E402
import frontier_svg as F  # noqa: E402

# categorical slots 1-4 (dataviz reference palette, light) then 5-8, fixed per trunk in this order
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a5cd0', '#d6457a', '#3d8f9e', '#7a6a4f']
INK, INK2, GRID, FRAME = '#0b0b0b', '#52514e', '#e5e5e3', '#a9a8a3'
W, H = 1180, 560
PL, PR, PT, PB = 70, 150, 70, 60          # left panel margins
GAP = 70
SPLIT = 520                               # left panel width (incl. margins)


def hitk_pct(name):
    try:
        cur = json.load(open(os.path.join(D.RUNS, name, 'curve.json')))
    except OSError:
        return None
    f = [x for x in cur['val'] if x.get('final')][-1]
    return 100 * f['buzz_hitk'] / f['buzz_teacher'] if 'buzz_hitk' in f else None


def trunks():
    """{trunk name: [rows ascending by steps]} for every comparable WSD branch in the ladder."""
    out = {}
    for r in LR.rows():            # comparable rows only (ladder_record.comparability)
        if r.get('schedule') != 'wsd' or r['headline'] != r['headline']:
            continue
        out.setdefault(r['decay_from'].split(':')[0], []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r['steps'])
        for r in out[k]:
            r['hitk'] = hitk_pct(r['name'])
    return {k: v for k, v in out.items() if len(v) > 1}     # a lone 7k branch (the equivalence runs) is no curve


def stop_index(rows, tol):
    """Index of the branch where the rule ends the trunk (the last one it runs)."""
    for i in range(1, len(rows)):
        p, c = rows[i - 1]['hitk'], rows[i]['hitk']
        if p is not None and c is not None and c - p < tol:
            return i
    return len(rows) - 1


def label(r):
    seed = f', seed {r["seed"]}' if r.get('seed', 1) != 1 else ''
    return f'{r["frontend"]} {r["arch"]}, rung {r["rung"]}{seed}'


def mark(fe, x, y, c, filled, ring=False, s=6):
    fill = c if filled else '#fff'
    sw = 2
    if fe == 'yamnet':
        m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s}"'
    else:
        m = f'<polygon points="{x:.1f},{y - s - 1.5:.1f} {x + s + 1.5:.1f},{y:.1f} {x:.1f},{y + s + 1.5:.1f} {x - s - 1.5:.1f},{y:.1f}"'
    out = f'{m} fill="{fill}" stroke="{c}" stroke-width="{sw}"/>'
    if ring:
        out = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s + 5}" fill="none" stroke="{INK}" stroke-width="1.5"/>' + out
    return out


def main():
    args = [a for a in sys.argv[1:]]
    tol = 1.85
    if '--tol' in args:
        i = args.index('--tol')
        tol = float(args[i + 1])
        del args[i:i + 2]
    out_path = args[0] if args else os.path.join(HERE, 'wsd.svg')
    tr = trunks()
    order = sorted(tr, key=lambda k: (tr[k][0]['frontend'] != 'yamnet', tr[k][0]['arch'] != 'a0.50', k))
    colour = {k: SLOT[i % len(SLOT)] for i, k in enumerate(order)}
    stop = {k: stop_index(tr[k], tol) for k in order}

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12" fill="{INK}">',
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
         f'<text x="{PL}" y="26" font-size="16" font-weight="bold">Step budget (WSD): sensitivity by training steps, and the frontier it moves</text>',
         f'<text x="{PL}" y="46" fill="{INK2}">buzz+rain+human students (seed 1 unless labelled); ringed = where --wsd-stop {tol:g} '
         f'(hit@K gain) ends the trunk; hollow = past that stop; headline seed noise ~0.01-0.02 (rung-A repeats)</text>']

    # ---- left: headline vs steps (log2)
    steps = sorted({r['steps'] for k in order for r in tr[k]})
    xl0, xl1 = math.log2(steps[0]) - 0.3, math.log2(steps[-1]) + 0.3
    allh = [r['headline'] for k in order for r in tr[k]]
    y0, y1 = math.floor(min(allh) * 20) / 20 - 0.01, math.ceil(max(allh) * 20) / 20 + 0.01
    x_a, x_b, y_a, y_b = PL, SPLIT - PR + 110, PT, H - PB
    px = lambda s: x_a + (math.log2(s) - xl0) / (xl1 - xl0) * (x_b - x_a - 110)
    py = lambda v: y_b - (v - y0) / (y1 - y0) * (y_b - y_a)
    v = y0 + (0.05 - y0 % 0.05) % 0.05
    while v <= y1 + 1e-9:
        o.append(f'<line x1="{x_a}" x2="{x_b - 110}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{x_a - 8}" y="{py(v) + 4:.1f}" text-anchor="end" fill="{INK2}">{v:.2f}</text>')
        v += 0.05
    for s in steps:
        o.append(f'<line x1="{px(s):.1f}" x2="{px(s):.1f}" y1="{y_a}" y2="{y_b}" stroke="{GRID}"/>'
                 f'<text x="{px(s):.1f}" y="{y_b + 18}" text-anchor="middle" fill="{INK2}">{s // 1000}k</text>')
    o.append(f'<line x1="{x_a}" x2="{x_b - 110}" y1="{y_b}" y2="{y_b}" stroke="{FRAME}"/>')
    o.append(f'<text x="{(x_a + x_b - 110) / 2}" y="{H - 16}" text-anchor="middle" fill="{INK2}">training steps (log scale)</text>')
    o.append(f'<text transform="translate(20 {(y_a + y_b) / 2}) rotate(-90)" text-anchor="middle" fill="{INK2}">'
             f'sensitivity (excl. quiet) @ 0.5% FPR</text>')
    ends = []
    for k in order:
        rows, c, si = tr[k], colour[k], stop[k]
        pts = [(px(r['steps']), py(r['headline'])) for r in rows]
        o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="'
                 + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts[:si + 1]) + '"/>')
        if si < len(rows) - 1:
            o.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="4 4" points="'
                     + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts[si:]) + '"/>')
        for i, (r, (x, y)) in enumerate(zip(rows, pts)):
            hk = f'{r["hitk"]:.2f}' if r['hitk'] is not None else 'n/a'
            o.append(f'<g>{mark(r["frontend"], x, y, c, i <= si, i == si)}<title>{r["name"]}\n'
                     f'headline {r["headline"]:.3f}  hit@K {hk}%  lost {r["lost_pct"]:.1f}%  mae_live {r["mae_live"]:.3f}</title></g>')
        ends.append([pts[-1][1], k, pts[-1][0]])
    ends.sort()                                    # direct labels at line ends, nudged apart
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    for y, k, x in ends:
        r = tr[k][0]
        o.append(f'<rect x="{x_b - 104}" y="{y - 5:.1f}" width="10" height="3" fill="{colour[k]}"/>'
                 f'<text x="{x_b - 90}" y="{y + 1:.1f}" font-size="11">{label(r)}</text>')

    # ---- right: speed frontier
    base = F.load()
    picks = [dict(tr[k][stop[k]], _k=k) for k in order]
    xa, xb, ya, yb = SPLIT + GAP, W - 30, PT, H - PB
    xs = [r['x_yamnet200'] for r in base + picks]
    fx0, fx1 = 0.8, max(xs) + 0.25
    fy0, fy1 = 0.40, 0.75
    qx = lambda v: xa + (v - fx0) / (fx1 - fx0) * (xb - xa)
    qy = lambda v: yb - (v - fy0) / (fy1 - fy0) * (yb - ya)
    for v in [0.4, 0.5, 0.6, 0.7]:
        o.append(f'<line x1="{xa}" x2="{xb}" y1="{qy(v):.1f}" y2="{qy(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{xa - 8}" y="{qy(v) + 4:.1f}" text-anchor="end" fill="{INK2}">{v:.1f}</text>')
    v = 1.0
    while v <= fx1:
        o.append(f'<line x1="{qx(v):.1f}" x2="{qx(v):.1f}" y1="{ya}" y2="{yb}" stroke="{GRID}"/>'
                 f'<text x="{qx(v):.1f}" y="{yb + 18}" text-anchor="middle" fill="{INK2}">{v:g}x</text>')
        v += 0.5
    o.append(f'<line x1="{xa}" x2="{xb}" y1="{yb}" y2="{yb}" stroke="{FRAME}"/>')
    o.append(f'<text x="{(xa + xb) / 2}" y="{H - 16}" text-anchor="middle" fill="{INK2}">speed, x YAMNet (GPU, 200 s audio)</text>')
    o.append(f'<line x1="{xa}" x2="{xb}" y1="{qy(F.BASELINE):.1f}" y2="{qy(F.BASELINE):.1f}" stroke="{INK2}" stroke-dasharray="4 3"/>'
             f'<text x="{xb - 4}" y="{qy(F.BASELINE) - 5:.1f}" text-anchor="end" fill="{INK2}">era baseline {F.BASELINE}</text>')
    for r in base:
        o.append(f'<circle cx="{qx(r["x_yamnet200"]):.1f}" cy="{qy(r["headline"]):.1f}" r="3.5" fill="#b8b7b1">'
                 f'<title>{r["name"]} (rung B cosine 7k)\nheadline {r["headline"]:.3f}  {r["x_yamnet200"]:.2f}x</title></circle>')
    old = F.pareto(base)
    o.append(f'<polyline fill="none" stroke="#b8b7b1" stroke-width="2" points="'
             + ' '.join(f'{qx(r["x_yamnet200"]):.1f},{qy(r["headline"]):.1f}' for r in old) + '"/>')
    new = F.pareto(base + picks)
    o.append(f'<polyline fill="none" stroke="{INK}" stroke-width="1.5" stroke-dasharray="2 3" points="'
             + ' '.join(f'{qx(r["x_yamnet200"]):.1f},{qy(r["headline"]):.1f}' for r in new) + '"/>')
    for k in order:
        c = colour[k]
        for i, r in enumerate(tr[k]):
            if i != stop[k]:
                o.append(f'<g>{mark(r["frontend"], qx(r["x_yamnet200"]), qy(r["headline"]), c, False, s=3.5)}'
                         f'<title>{r["name"]}\nheadline {r["headline"]:.3f}</title></g>')
    for p in picks:
        x, y = qx(p['x_yamnet200']), qy(p['headline'])
        o.append(f'<g>{mark(p["frontend"], x, y, colour[p["_k"]], True, True)}<title>{p["name"]} (stop-rule branch)\n'
                 f'headline {p["headline"]:.3f}  {p["x_yamnet200"]:.2f}x  {p["steps"] // 1000}k steps</title></g>'
                 f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-size="11">{p["steps"] // 1000}k</text>')
    lx, ly = xa + 12, qy(0.505)
    o.append(f'<line x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}" stroke="#b8b7b1" stroke-width="2"/>'
             f'<text x="{lx + 24}" y="{ly + 4}" font-size="11">rung B, cosine 7k: runs and frontier</text>'
             f'<line x1="{lx}" x2="{lx + 18}" y1="{ly + 18}" y2="{ly + 18}" stroke="{INK}" stroke-dasharray="2 3" stroke-width="1.5"/>'
             f'<text x="{lx + 24}" y="{ly + 22}" font-size="11">frontier with the stop-rule branches</text>'
             f'<circle cx="{lx + 9}" cy="{ly + 38}" r="5" fill="#fff" stroke="{INK2}" stroke-width="2"/>'
             f'<text x="{lx + 24}" y="{ly + 42}" font-size="11">YAMNet front end</text>'
             f'<polygon points="{lx + 9},{ly + 50} {lx + 16},{ly + 57} {lx + 9},{ly + 64} {lx + 2},{ly + 57}" fill="#fff" stroke="{INK2}" stroke-width="2"/>'
             f'<text x="{lx + 24}" y="{ly + 61}" font-size="11">fast32* front end</text>')
    o.append('</svg>')
    open(out_path, 'w').write('\n'.join(o))
    print(out_path)
    for k in order:
        rows = tr[k]
        print(f'  {label(rows[0]):28s} ' + '  '.join(
            f'{r["steps"] // 1000}k {r["headline"]:.3f}/{r["hitk"] if r["hitk"] is None else round(r["hitk"], 2)}'
            + ('*' if i == stop[k] else '') for i, r in enumerate(rows)))


if __name__ == '__main__':
    main()
