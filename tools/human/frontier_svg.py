"""Frontier scatter as a standalone SVG: every seed-1 buzz+rain+human student with a speed, all rungs, one point
per (student, rung): a WSD curve's branch at the --wsd-stop rule, else its 7k cosine run. Clean rows (the live
log) and the quarantined contaminated_2026-10-02 rows are drawn alike (informative). x = speed (x YAMNet, GPU,
200 s), y = headline (sensitivity_exclquiet @ fpr 0.005). Colour = trunk architecture, marker = front end,
outline = rung (none A, thin B, thick C, double D). A thin grey line joins one student across rungs; "28k"
marks a budget other than 7k.

Below it, a bar chart of the jet fold: for each non-dominated student, 1_95's false-positive rate when the pooled
rotating-fold threshold (fpr 0.005 over every rotating fold's negatives) is applied to it, on all its negatives and on its
jet frames alone, against the rotating folds' own worst FPR at that threshold (eval_folds.py probe -> probe.json).

    python tools/human/frontier_svg.py [out.svg]      (train env; reads the teacher's comparable rows of 05_distill/log.jsonl)
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '05_distill'))
import dpaths as D  # noqa: E402
import ladder_record as LR  # noqa: E402

BASELINE = 0.414
ARCH_COLOUR = {'a0.25': '#d95f02', 'a0.375': '#7570b3', 'a0.50': '#1b9e77', 'a0.50_d12': '#e7298a'}
FE_MARK = {'yamnet': 'circle', 'fast32': 'square', 'fast32h16': 'diamond', 'fast32h32': 'tri',
           'twofast32': 'cross', 'two32': 'hex', 'lo32': 'hex', 'fast32lo': 'square', 'fast32h16lo': 'diamond'}
W, H, L, R, T, B = 960, 680, 70, 220, 40, 60
BRH = 'ins_buzz,ambient_rain,human'
RUNG_OUTLINE = {'A': 0, 'B': 1.2, 'C': 3, 'D': 3}      # stroke width; D adds an outer ring
H2 = 340        # bar chart panel under the scatter


def load():
    out = []
    for r in LR.rows():            # comparable rows only (ladder_record.comparability)
        if r.get('rung') != 'B' or r.get('seed') != 1 or not r.get('x_yamnet200'):
            continue
        if r.get('schedule', 'cosine') != 'cosine':     # WSD branches belong to `ladder_record.py wsd`
            continue
        if r['headline'] != r['headline']:
            continue
        if r.get('init') == '' and r['name'].endswith('_stream'):
            continue
        r.setdefault('frontend', 'yamnet')
        out.append(r)
    return out


def run_file(name, kind, *parts):
    """<kind>/<name>/... in the live data dir, else in a quarantine dir (<data>/_<tag>/<kind>/<name>), else None."""
    for root in [D.LOCAL] + sorted(glob.glob(os.path.join(D.LOCAL, '_*'))):
        p = os.path.join(root, kind, name, *parts)
        if os.path.exists(p):
            return p
    return None


def ident(r):
    """One student on one rung: everything but seed, steps and schedule."""
    return (r['frontend'], r['arch'], r.get('init') or '', r.get('classes') or '',
            0.1 if r.get('lam') is None else r['lam'], r['rung'])


def hitk(r):
    if r.get('hitk_pct') is not None:
        return r['hitk_pct']
    p = run_file(r['name'], 'runs', 'curve.json')
    v = [x for x in json.load(open(p))['val'] if x.get('final')][-1] if p else {}
    return 100 * v['buzz_hitk'] / max(1, v['buzz_teacher']) if 'buzz_hitk' in v else None


def load_all(tol=None):
    """Every seed-1 student with a speed, live log (pool 'clean') plus the tagged logs beside it (pool = tag),
    one point per ident(): a WSD curve's stop-rule branch (main.py's `plateaued`), else its cosine run."""
    tol = tol or LR.hitk_spread() or 1.85
    raw = [dict(r, pool='clean') for r in LR.rows()]
    for tag, f in D.tagged_logs().items():
        raw += [dict(r, pool=tag) for r in D.read_log(f)]
    curves, pts = {}, {}
    for r in raw:
        if r.get('seed') != 1 or not r.get('x_yamnet200') or r.get('headline') is None \
                or r['headline'] != r['headline'] or r['name'].endswith('_stream'):
            continue
        r['frontend'], r['arch'] = r.get('frontend') or 'yamnet', r.get('arch') or 'a0.50'
        if r.get('schedule', 'cosine') == 'wsd':
            curves.setdefault(r['decay_from'].split(':')[0], []).append(r)
        else:
            pts[ident(r)] = r
    for rs in curves.values():
        if len(rs) < 2:                      # a lone 7k branch (the equivalence runs) duplicates the cosine run
            continue
        rs.sort(key=lambda r: r['steps'])
        hk = [hitk(r) for r in rs]
        i = next((j for j in range(1, len(rs)) if None not in (hk[j - 1], hk[j]) and hk[j] - hk[j - 1] < tol),
                 len(rs) - 1)
        pts[ident(rs[i])] = rs[i]
    return list(pts.values())


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
        return json.load(open(run_file(name, 'eval', 'probe', 'probe.json')))[0].get('cross')
    except (OSError, KeyError, IndexError, TypeError):
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
             f'<text x="{W - R + 6}" y="{py(0.005) + 4}" fill="#555">target 0.5%</text>')
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
        lab = r['frontend']
        o.append(f'<text x="{cx:.1f}" y="{bot + 14}" font-size="9" text-anchor="middle">{lab}</text>'
                 f'<text x="{cx:.1f}" y="{bot + 26}" font-size="9" text-anchor="middle" fill="#555">{r["arch"]} · '
                 f'{r.get("rung", "B")} · {r["x_yamnet200"]:.1f}x</text>')
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


def marker(kind, x, y, c, rung=None, s=7, alpha=0.85):
    """One point: shape = front end, fill = trunk colour, outline = rung (RUNG_OUTLINE)."""
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
    sw = RUNG_OUTLINE.get(rung, 0)
    out = f'{m} fill="{c}" fill-opacity="{alpha}" stroke="{"#111" if sw else "none"}" stroke-width="{sw}"/>'
    if rung == 'D':
        out += f'<circle cx="{x}" cy="{y}" r="{s + 5}" fill="none" stroke="#111" stroke-width="1.2"/>'
    return out


def label(r):
    lab = f'{r["frontend"]} {r["arch"]}'
    return lab + f' · {r["rung"]}' + (f' {r["steps"] // 1000}k' if r['steps'] != 7000 else '')


def main():
    rows = [r for r in load_all() if r.get('classes') == BRH]
    if not rows:
        sys.exit('no seed-1 buzz+rain+human students with a speed in the ladder yet')
    xs = [r['x_yamnet200'] for r in rows]
    x0, x1 = 0.6, max(xs) + 0.3
    y0, y1 = 0.35, 0.75
    px = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    py = lambda v: H - B - (v - y0) / (y1 - y0) * (H - B - T)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + H2}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H + H2}" fill="#fff"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="bold">Speed / sensitivity frontier: buzz+rain+human '
         f'students, every rung, seed 1, each at its stop-rule budget</text>']
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
    # one student across rungs: a thin line from its smaller rung to its larger
    by = {}
    for r in rows:
        by.setdefault(ident(r)[:-1], []).append(r)
    for rs in by.values():
        if len(rs) > 1:
            rs.sort(key=lambda r: r['rung'])
            o.append('<polyline fill="none" stroke="#9a9a9a" stroke-width="1.2" points="'
                     + ' '.join(f'{px(r["x_yamnet200"]):.1f},{py(r["headline"]):.1f}' for r in rs) + '"/>')
    fr = pareto(rows)
    o.append('<polyline fill="none" stroke="#111" stroke-width="1.5" stroke-dasharray="2 3" points="'
             + ' '.join(f'{px(r["x_yamnet200"]):.1f},{py(r["headline"]):.1f}' for r in fr) + '"/>')
    placed = []                                  # label boxes (x0, x1, y), pushed apart vertically

    def put_label(x, y, text):
        w = 5.1 * len(text)
        left = x + 11 + w > W - R - 4
        x0 = x - 11 - w if left else x + 11
        ly = y - 9
        while any(abs(ly - py_) < 12 and x0 < b1 and x0 + w > a0 for a0, b1, py_ in placed):
            ly += 12
        placed.append((x0, x0 + w, ly))
        for halo in ('stroke="#fff" stroke-width="3" stroke-linejoin="round" ', ''):   # white halo, then the text
            o.append(f'<text x="{x0:.1f}" y="{ly:.1f}" font-size="10" {halo}>{text}</text>')

    for r in sorted(rows, key=lambda r: r['rung']):                      # larger rungs on top
        x, y = px(r['x_yamnet200']), py(r['headline'])
        c = ARCH_COLOUR.get(r['arch'], '#888')
        o.append(f'<g>{marker(FE_MARK.get(r["frontend"], "circle"), round(x, 1), round(y, 1), c, r["rung"])}'
                 f'<title>{r["name"]} ({r["pool"]})\nheadline {r["headline"]:.3f}  {r["x_yamnet200"]:.2f}x  '
                 f'{r["steps"] // 1000}k steps</title></g>')
        if r['steps'] != 7000 and r not in fr:
            o.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="9" fill="#555">{r["steps"] // 1000}k</text>')
    for r in sorted(fr, key=lambda r: py(r['headline'])):
        put_label(px(r['x_yamnet200']), py(r['headline']), label(r))
    lx, ly = W - R + 20, T + 10
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Trunk</text>')
    archs = [(a, c) for a, c in ARCH_COLOUR.items() if any(r['arch'] == a for r in rows)]   # only what is drawn
    for i, (a, c) in enumerate(archs):
        o.append(f'<rect x="{lx}" y="{ly + 8 + i * 18}" width="12" height="12" fill="{c}"/>'
                 f'<text x="{lx + 18}" y="{ly + 18 + i * 18}">{a}</text>')
    ly += 18 * len(archs) + 28
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Front end</text>')
    shapes = {FE_MARK.get(r['frontend'], 'circle') for r in rows}
    fes = [(k, lab) for k, lab in [('circle', 'YAMNet'), ('square', 'fast32 / lo'), ('diamond', 'fast32h16'),
                                   ('tri', 'fast32h32'), ('cross', 'twofast32'), ('hex', 'two32 / lo32')] if k in shapes]
    for i, (k, lab) in enumerate(fes):
        o.append(marker(k, lx + 6, ly + 16 + i * 18, '#888')
                 + f'<text x="{lx + 22}" y="{ly + 20 + i * 18}">{lab}</text>')
    ly += 18 * len(fes) + 40
    o.append(f'<text x="{lx}" y="{ly - 14}" font-weight="bold">Rung (training data)</text>')
    rungs = sorted({r['rung'] for r in rows})
    for i, k in enumerate(rungs):
        o.append(marker('circle', lx + 8, ly + 4 + i * 22, '#bbb', k)
                 + f'<text x="{lx + 26}" y="{ly + 8 + i * 22}">{ {"A": "A: smallest", "B": "B: 4x A", "C": "C: 4x B", "D": "D: all"}[k]}</text>')
    ly += 22 * len(rungs) + 14
    o.append(f'<line x1="{lx}" x2="{lx + 16}" y1="{ly}" y2="{ly}" stroke="#9a9a9a" stroke-width="1.2"/>'
             f'<text x="{lx + 26}" y="{ly + 4}">same student, B -> C</text>'
             f'<line x1="{lx}" x2="{lx + 16}" y1="{ly + 20}" y2="{ly + 20}" stroke="#111" stroke-dasharray="2 3"/>'
             f'<text x="{lx + 26}" y="{ly + 24}">non-dominated</text>'
             f'<text x="{lx}" y="{ly + 44}" font-size="10" fill="#555">28k = steps, if not 7k</text>')
    bars(o, fr)
    o.append('</svg>')
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontier.svg')
    open(out, 'w').write('\n'.join(o))
    print(out, len(rows), 'students,', len(fr), 'on the frontier,',
          sum(r['pool'] == 'clean' for r in rows), 'clean')


if __name__ == '__main__':
    main()
