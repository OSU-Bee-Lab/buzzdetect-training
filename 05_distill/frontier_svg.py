"""Frontier scatter as a standalone SVG: rung-B seed-1 runs, x = speed (x YAMNet, GPU, 200 s), y = headline
(sensitivity_exclquiet @ fpr 0.005). Colour = trunk architecture, marker = front end, ring = class-subset student (plain ring: ins_buzz+rain+human,
ring with a centre dot: ins_buzz alone).

Below it, a bar chart of the jet fold: for each non-dominated student, 1_95's false-positive rate when the pooled
rotating-fold threshold (fpr 0.005 over every rotating fold's negatives) is applied to it, on all its negatives and on its
jet frames alone, against the rotating folds' own worst FPR at that threshold (eval_folds.py probe -> probe.json).

    python 05_distill/frontier_svg.py [out.svg]      (plain python; reads the per-teacher ladder.jsonl)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpaths as D  # noqa: E402

BASELINE = 0.414
ARCH_COLOUR = {'a0.25': '#d95f02', 'a0.375': '#7570b3', 'a0.50': '#1b9e77', 'a0.50_d12': '#e7298a'}
FE_MARK = {'yamnet': 'circle', 'fast32': 'square', 'fast32h16': 'diamond', 'fast32h32': 'tri',
           'twofast32': 'cross', 'two32': 'hex', 'lo32': 'hex', 'fast32lo': 'square', 'fast32h16lo': 'diamond'}
W, H, L, R, T, B = 900, 560, 70, 200, 40, 60
H2 = 340        # bar chart panel under the scatter


def load():
    out = []
    for line in open(os.path.join(D.LOCAL, 'ladder.jsonl')):
        r = json.loads(line)
        if r.get('rung') != 'B' or r.get('seed') != 1 or not r.get('x_yamnet200'):
            continue
        if r['headline'] != r['headline']:
            continue
        if r.get('init') == '' and r['name'].endswith('_stream'):
            continue
        r.setdefault('frontend', 'yamnet')
        out.append(r)
    return out


def pareto(rows):
    pts = sorted(rows, key=lambda r: -r['x_yamnet200'])
    best, front = -1, []
    for r in pts:
        if r['headline'] > best:
            front.append(r)
            best = r['headline']
    return front


def probe_of(name):
    try:
        return json.load(open(os.path.join(D.EVAL, name, 'probe', 'probe.json')))[0].get('cross')
    except (OSError, KeyError, IndexError):
        return None


def bars(o, fr):
    """Second panel: 1_95 FPR at the pooled rotating-fold threshold, one group per frontier student (fast to slow)."""
    items = [(r, probe_of(r['name'])) for r in sorted(fr, key=lambda r: -r['x_yamnet200'])]
    items = [(r, c['pooled']) for r, c in items if c]
    top, bot = H + 50, H + H2 - 70
    ymax = 0.03
    py = lambda v: bot - v / ymax * (bot - top)
    o.append(f'<text x="{L}" y="{H + 12}" font-size="15" font-weight="bold">Jet fold 1_95: false-positive rate at the '
             f"rotating folds' threshold</text>")
    o.append(f'<text x="{L}" y="{H + 30}" fill="#555">pooled fpr-0.005 threshold from the five rotating folds, applied to '
             f'1_95 (a training fold); jets = frames labelled mech_plane</text>')
    for v in [0, 0.005, 0.01, 0.02, 0.03]:
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(v)}" y2="{py(v)}" stroke="#e5e5e5"/>'
                 f'<text x="{L - 8}" y="{py(v) + 4}" text-anchor="end">{v:.1%}</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(0.005)}" y2="{py(0.005)}" stroke="#555" stroke-dasharray="4 3"/>'
             f'<text x="{W - R - 4}" y="{py(0.005) - 5}" text-anchor="end" fill="#555">target 0.5%</text>')
    o.append(f'<text transform="translate(18 {(top + bot) / 2}) rotate(-90)" text-anchor="middle">1_95 false-positive rate</text>')
    if not items:
        o.append(f'<text x="{L + 20}" y="{(top + bot) / 2}">no probe.json yet: run eval_folds.py probe</text>')
        return
    gw = (W - L - R) / len(items)
    bw = min(26, gw / 3.2)
    for i, (r, c) in enumerate(items):
        cx = L + gw * (i + 0.5)
        col = ARCH_COLOUR.get(r['arch'], '#888')
        for dx, key, fill in ((-bw - 1, 'fpr', col), (1, 'fpr_jet', '#111')):
            v = min(c[key], ymax)
            o.append(f'<rect x="{cx + dx:.1f}" y="{py(v):.1f}" width="{bw:.1f}" height="{bot - py(v):.1f}" fill="{fill}" '
                     f'fill-opacity="0.85"><title>{r["name"]} {key} {c[key]:.2%}</title></rect>'
                     f'<text x="{cx + dx + bw / 2:.1f}" y="{py(v) - 4:.1f}" font-size="9" text-anchor="middle">'
                     f'{c[key] * 100:.1f}</text>')
        o.append(f'<line x1="{cx - bw - 4:.1f}" x2="{cx + bw + 4:.1f}" y1="{py(c["rotating_fpr_max"]):.1f}" '
                 f'y2="{py(c["rotating_fpr_max"]):.1f}" stroke="#999" stroke-width="3"/>')
        lab = r['frontend'] + SUB_LABEL[subset(r)]
        o.append(f'<text x="{cx:.1f}" y="{bot + 14}" font-size="10" text-anchor="middle">{lab}</text>'
                 f'<text x="{cx:.1f}" y="{bot + 27}" font-size="10" text-anchor="middle" fill="#555">{r["arch"]}, '
                 f'{r["x_yamnet200"]:.1f}x</text>')
    o.append(f'<rect x="{L}" y="{top}" width="{W - L - R}" height="{bot - top}" fill="none" stroke="#999"/>')
    lx, ly = W - R + 20, top + 10
    for k, (fill, lab) in enumerate([('#888', '1_95, all negatives'), ('#111', '1_95, jet frames only')]):
        o.append(f'<rect x="{lx}" y="{ly + k * 18}" width="12" height="12" fill="{fill}"/>'
                 f'<text x="{lx + 18}" y="{ly + 10 + k * 18}" font-size="11">{lab}</text>')
    o.append(f'<line x1="{lx}" x2="{lx + 12}" y1="{ly + 42}" y2="{ly + 42}" stroke="#999" stroke-width="3"/>'
             f'<text x="{lx + 18}" y="{ly + 46}" font-size="11">worst rotating fold</text>')


def subset(r):
    """'' for a full 15-class student, 'buzz' for ins_buzz alone, else 'sub' (the three-class subset)."""
    k = r.get('classes')
    if not k:
        return ''
    return 'buzz' if k == 'ins_buzz' else 'sub'


SUB_LABEL = {'': '', 'sub': ' +3cls', 'buzz': ' +buzz'}


def marker(kind, x, y, c, ring):
    s = 7
    if kind == 'square':
        m = f'<rect x="{x - s}" y="{y - s}" width="{2 * s}" height="{2 * s}"'
    elif kind == 'diamond':
        m = f'<polygon points="{x},{y - s - 2} {x + s + 2},{y} {x},{y + s + 2} {x - s - 2},{y}"'
    elif kind == 'tri':
        m = f'<polygon points="{x},{y - s - 1} {x + s + 1},{y + s} {x - s - 1},{y + s}"'
    elif kind == 'hex':
        m = f'<polygon points="{x - s},{y} {x - s / 2},{y - s} {x + s / 2},{y - s} {x + s},{y} {x + s / 2},{y + s} {x - s / 2},{y + s}"'
    elif kind == 'cross':
        m = (f'<polygon points="{x - 2.5},{y - s} {x + 2.5},{y - s} {x + 2.5},{y - 2.5} {x + s},{y - 2.5} {x + s},{y + 2.5} '
             f'{x + 2.5},{y + 2.5} {x + 2.5},{y + s} {x - 2.5},{y + s} {x - 2.5},{y + 2.5} {x - s},{y + 2.5} {x - s},{y - 2.5} '
             f'{x - 2.5},{y - 2.5}"')
    else:
        m = f'<circle cx="{x}" cy="{y}" r="{s}"'
    stroke = '#111' if ring else 'none'
    sw = 2.5 if ring else 0
    out = f'{m} fill="{c}" fill-opacity="0.85" stroke="{stroke}" stroke-width="{sw}"/>'
    if ring == 'buzz':
        out += f'<circle cx="{x}" cy="{y}" r="2.2" fill="#111"/>'
    return out


def main():
    rows = load()
    xs = [r['x_yamnet200'] for r in rows]
    x0, x1 = 0.6, max(xs) + 0.3
    y0, y1 = 0.35, 0.75
    px = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    py = lambda v: H - B - (v - y0) / (y1 - y0) * (H - B - T)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + H2}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H + H2}" fill="#fff"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="bold">Speed / sensitivity frontier (rung B, seed 1)</text>']
    for v in [0.4, 0.5, 0.6, 0.7]:
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(v)}" y2="{py(v)}" stroke="#e5e5e5"/>'
                 f'<text x="{L - 8}" y="{py(v) + 4}" text-anchor="end">{v:.1f}</text>')
    for v in [1, 1.5, 2, 2.5, 3]:
        o.append(f'<line y1="{T}" y2="{H - B}" x1="{px(v)}" x2="{px(v)}" stroke="#e5e5e5"/>'
                 f'<text x="{px(v)}" y="{H - B + 18}" text-anchor="middle">{v:g}x</text>')
    o.append(f'<line x1="{px(1)}" x2="{px(1)}" y1="{T}" y2="{H - B}" stroke="#555" stroke-dasharray="4 3"/>'
             f'<text x="{px(1) + 4}" y="{T + 12}" fill="#555">YAMNet speed</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(BASELINE)}" y2="{py(BASELINE)}" stroke="#555" stroke-dasharray="4 3"/>'
             f'<text x="{W - R - 4}" y="{py(BASELINE) - 5}" text-anchor="end" fill="#555">era baseline 0.414</text>')
    o.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - B - T}" fill="none" stroke="#999"/>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle">speed, x YAMNet (GPU, 200 s audio; higher is faster)</text>')
    o.append(f'<text transform="translate(18 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">'
             f'sensitivity (excl. quiet) @ 0.5% FPR</text>')
    fr = pareto(rows)
    o.append('<polyline fill="none" stroke="#111" stroke-width="1.5" stroke-dasharray="2 3" points="'
             + ' '.join(f'{px(r["x_yamnet200"]):.1f},{py(r["headline"]):.1f}' for r in fr) + '"/>')
    for r in rows:
        x, y = px(r['x_yamnet200']), py(r['headline'])
        c = ARCH_COLOUR.get(r['arch'], '#888')
        o.append(marker(FE_MARK.get(r['frontend'], 'circle'), round(x, 1), round(y, 1), c, subset(r)))
        if r in fr:
            lab = r['frontend'] + SUB_LABEL[subset(r)]
            o.append(f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" font-size="10">{lab}</text>')
    lx, ly = W - R + 20, T + 10
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Trunk</text>')
    for i, (a, c) in enumerate(ARCH_COLOUR.items()):
        o.append(f'<rect x="{lx}" y="{ly + 8 + i * 18}" width="12" height="12" fill="{c}"/>'
                 f'<text x="{lx + 18}" y="{ly + 18 + i * 18}">{a}</text>')
    ly += 100
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Front end</text>')
    for i, (k, lab) in enumerate([('circle', 'YAMNet'), ('square', 'fast32 / lo'), ('diamond', 'fast32h16'),
                                  ('tri', 'fast32h32'), ('cross', 'twofast32'), ('hex', 'two32 / lo32')]):
        o.append(marker(k, lx + 6, ly + 16 + i * 18, '#888', False)
                 + f'<text x="{lx + 22}" y="{ly + 20 + i * 18}">{lab}</text>')
    ly += 142
    o.append(f'<text x="{lx}" y="{ly - 14}" font-weight="bold">Classes</text>')
    o.append(marker('circle', lx + 6, ly + 2, '#fff', '') + f'<text x="{lx + 22}" y="{ly + 6}">all 15</text>')
    o.append(marker('circle', lx + 6, ly + 20, '#fff', 'sub') + f'<text x="{lx + 22}" y="{ly + 24}">buzz + rain + human</text>')
    o.append(marker('circle', lx + 6, ly + 38, '#fff', 'buzz') + f'<text x="{lx + 22}" y="{ly + 42}">buzz only</text>')
    o.append(f'<line x1="{lx}" x2="{lx + 12}" y1="{ly + 60}" y2="{ly + 60}" stroke="#111" stroke-dasharray="2 3"/>'
             f'<text x="{lx + 22}" y="{ly + 64}">non-dominated</text>')
    bars(o, fr)
    o.append('</svg>')
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontier.svg')
    open(out, 'w').write('\n'.join(o))
    print(out, len(rows), 'runs,', len(fr), 'on the frontier')


if __name__ == '__main__':
    main()
