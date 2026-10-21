"""Frontier scatter as a standalone SVG: every seed-1 buzz+rain+human student with a speed, all rungs, one point
per (student, rung): a WSD curve's branch at the --wsd-stop rule, else its 7k cosine run. Clean rows (the live
log) and the quarantined contaminated_2026-10-02 rows are both drawn, the quarantined ones faded. x = speed
(x YAMNet, GPU, 200 s), y = headline (sensitivity_exclquiet @ fpr 0.005). Colour = trunk width, marker = front-end
family, outline = rung (none A, thin B, thick C); a dashed outline is a speed timed only before the mmt export.
A thin grey line joins one student across rungs.

Speed belongs to the graph, not the run: a re-export (`_mmt`, `_mmtz`, `_cfe`: same weights, another graph, the same
headline; 05_distill/CLAUDE.md) folds into its run, and every run of one graph (front end, trunk, classes) takes that
graph's fastest timed export. Speeds timed before 2026-10-11 include a Pad that costs more than a fast student's
whole run, so without this the frontier mixed two timing regimes.

Below it, a bar chart of the jet fold: for each non-dominated student, 1_95's false-positive rate when the pooled
rotating-fold threshold (fpr 0.005 over every rotating fold's negatives) is applied to it, on all its negatives and on its
jet frames alone, against the rotating folds' own worst FPR at that threshold (eval_folds.py probe -> probe.json).

    python tools/human/frontier_svg.py [out.svg]      (train env; reads the teacher's comparable rows of 05_distill/log.jsonl)
"""
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '05_distill'))
import dpaths as D  # noqa: E402
import ladder_record as LR  # noqa: E402

BASELINE = 0.414
BRH = 'ins_buzz,ambient_rain,human'
REEXPORT = re.compile(r'_(mmtz?|cfe)$')        # same weights, another graph: the headline is identical
PRE_MMT = ('stft', 'cfe')                      # exports whose timing includes the 200 s Pad
WIDTH_COLOUR = {'0.25': '#d95f02', '0.375': '#7570b3', '0.50': '#1b9e77', '0.75': '#e7298a'}
SPARE_COLOUR = ['#66a61e', '#e6ab02', '#a6761d', '#1f78b4']
# front-end families, first match wins: a new member (yam64h40, fast32h16lo) takes its family's glyph
FAMILIES = [('YAMNet', r'^yamnet$', 'circle'), ('yam64 (64 mel)', r'^yam64', 'tridown'), ('b64', r'^b64', 'star'),
            ('twofast32', r'^twofast32', 'cross'), ('fast32h32', r'^fast32h32', 'tri'),
            ('fast32h16', r'^fast32h16', 'diamond'), ('fast32 / lo', r'^fast32', 'square'),
            ('two32 / lo32', r'^(two32|lo32)', 'hex')]
RUNG_OUTLINE = {'A': 0, 'B': 1.2, 'C': 2.6, 'D': 2.6}      # stroke width
RUNG_NAME = {'A': 'A: smallest', 'B': 'B: 4x A', 'C': 'C: 4x B', 'D': 'D: all'}
INK, INK2, GRID, BOX = '#1a1a1a', '#555', '#e8e8e8', '#999'
W, H, L, R, T, B = 1000, 820, 70, 230, 56, 56
H2 = 330        # bar chart panel under the scatter


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


def graph(r):
    """What a speed measures: the exported graph, whatever trained its weights."""
    return r['frontend'], r['arch'], r.get('classes') or ''


def hitk(r):
    if r.get('hitk_pct') is not None:
        return r['hitk_pct']
    p = run_file(r['name'], 'runs', 'curve.json')
    v = [x for x in json.load(open(p))['val'] if x.get('final')][-1] if p else {}
    return 100 * v['buzz_hitk'] / max(1, v['buzz_teacher']) if 'buzz_hitk' in v else None


def fold_exports(raw):
    """One row per trained run (pool, name without a re-export suffix), carrying its fastest export's speed."""
    runs = {}
    for r in raw:
        runs.setdefault((r['pool'], REEXPORT.sub('', r['name'])), []).append(r)
    out = []
    for (_, base), rs in runs.items():
        timed = [x for x in rs if x.get('x_yamnet200')]
        fast = max(timed, key=lambda x: x['x_yamnet200']) if timed else None
        r = dict(next((x for x in rs if x['name'] == base), rs[0]), name=base)
        if fast:
            m = REEXPORT.search(fast['name'])
            r.update(x_yamnet200=fast['x_yamnet200'], x_yamnet20=fast.get('x_yamnet20'),
                     export=m.group(1) if m else 'stft', speed_from=fast['name'])
        out.append(r)
    return out


def load_all(tol=None):
    """Every seed-1 student with a speed, live log (pool 'clean') plus the tagged logs beside it (pool = tag),
    one point per ident(): a WSD curve's stop-rule branch (`dpaths.wsd_stop_index`), else its cosine run.
    Speed: the fastest timed export of the point's graph(), over every run and pool (module docstring)."""
    tol = tol or D.WSD_STOP
    raw = [dict(r, pool='clean') for r in LR.rows()]
    for tag, f in D.tagged_logs().items():
        raw += [dict(r, pool=tag) for r in D.read_log(f)]
    raw = [r for r in fold_exports(raw) if r.get('seed') == 1 and r.get('headline') is not None
           and r['headline'] == r['headline'] and not r['name'].endswith('_stream')]
    for r in raw:
        r['frontend'], r['arch'] = r.get('frontend') or 'yamnet', r.get('arch') or 'a0.50'
    best = {}
    for r in raw:
        if r.get('x_yamnet200') and r['x_yamnet200'] > best.get(graph(r), {}).get('x_yamnet200', 0):
            best[graph(r)] = r
    curves, pts = {}, {}
    for r in raw:
        if graph(r) not in best:             # a graph nobody timed has no place on a speed axis
            continue
        if r.get('schedule', 'cosine') == 'wsd':
            curves.setdefault(r['decay_from'].split(':')[0], []).append(r)
        else:
            pts[ident(r)] = r
    for rs in curves.values():
        if len(rs) < 2:                      # a lone 7k branch (the equivalence runs) duplicates the cosine run
            continue
        rs.sort(key=lambda r: r['steps'])
        hk = [hitk(r) for r in rs]
        i = D.wsd_stop_index(hk, tol)
        i = len(rs) - 1 if i is None else i
        pts[ident(rs[i])] = rs[i]
    out = []
    for r in pts.values():
        f = best[graph(r)]
        if f is not r and f['x_yamnet200'] > (r.get('x_yamnet200') or 0):
            r = dict(r, x_yamnet200=f['x_yamnet200'], x_yamnet20=f.get('x_yamnet20'), export=f['export'],
                     speed_from=f['speed_from'])
        out.append(r)
    return out


def pareto(rows):
    pts = sorted(rows, key=lambda r: (-r['x_yamnet200'], -r['headline']))
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


def subset(r):
    """'' for a full 15-class student, 'buzz' for ins_buzz alone, else 'sub' (the three-class subset)."""
    k = r.get('classes')
    if not k:
        return ''
    return 'buzz' if k == 'ins_buzz' else 'sub'


def trunk(arch):
    """'a<width>[_d<depth>][_<variant>...]' -> (width, depth, variant); depth 14 is the full YAMNet stack."""
    m = re.match(r'a([\d.]+)(?:_d(\d+))?(.*)$', arch or '')
    if not m:
        return '?', 14, ''
    w = m.group(1)
    return f'{float(w):.{3 if len(w) > 4 else 2}f}', int(m.group(2) or 14), m.group(3).lstrip('_')


def width_colours(rows):
    cols, spare = dict(WIDTH_COLOUR), iter(SPARE_COLOUR * 4)
    for w in sorted({trunk(r['arch'])[0] for r in rows}):
        cols.setdefault(w, next(spare))
    return cols


def family(frontend):
    return next((f for f in FAMILIES if re.search(f[1], frontend)), ('other', '', 'circle'))


def pre_mmt(r):
    return r.get('export', 'stft') in PRE_MMT


def marker(kind, x, y, c, rung=None, s=6.5, faded=False, dashed=False):
    """One point: shape = front-end family, fill = width colour, outline = rung (RUNG_OUTLINE)."""
    def poly(p):
        return '<polygon points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in p) + '"'
    if kind == 'square':
        m = f'<rect x="{x - s + .5:.1f}" y="{y - s + .5:.1f}" width="{2 * s - 1}" height="{2 * s - 1}"'
    elif kind == 'diamond':
        m = poly([(x, y - s - 2), (x + s + 2, y), (x, y + s + 2), (x - s - 2, y)])
    elif kind == 'tri':
        m = poly([(x, y - s - 1.5), (x + s + 1.5, y + s), (x - s - 1.5, y + s)])
    elif kind == 'tridown':
        m = poly([(x, y + s + 1.5), (x + s + 1.5, y - s), (x - s - 1.5, y - s)])
    elif kind == 'hex':
        m = poly([(x - s, y), (x - s / 2, y - s), (x + s / 2, y - s), (x + s, y), (x + s / 2, y + s), (x - s / 2, y + s)])
    elif kind == 'star':
        m = poly([(x + (s * .5 if i % 2 else s + 2.5) * math.cos(math.pi * i / 5 - math.pi / 2),
                   y + (s * .5 if i % 2 else s + 2.5) * math.sin(math.pi * i / 5 - math.pi / 2)) for i in range(10)])
    elif kind == 'cross':
        a = 2.5
        m = poly([(x - a, y - s), (x + a, y - s), (x + a, y - a), (x + s, y - a), (x + s, y + a), (x + a, y + a),
                  (x + a, y + s), (x - a, y + s), (x - a, y + a), (x - s, y + a), (x - s, y - a), (x - a, y - a)])
    else:
        m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s}"'
    sw = RUNG_OUTLINE.get(rung, 0)
    stroke = INK if sw else (c if faded else 'none')
    sw = sw or (1 if faded else 0)
    if dashed:
        stroke, sw = INK, max(sw, 1.4)
    return (f'{m} fill="{c}" fill-opacity="{.3 if faded else .9}" stroke="{stroke}" stroke-width="{sw}"'
            + (f' stroke-opacity="{.5 if faded else 1}"' if stroke == INK else '')
            + (' stroke-dasharray="2.5 1.8"' if dashed else '') + '/>')


def label(r):
    w, d, v = trunk(r['arch'])
    return (f'{r["frontend"]} a{w}' + (f' d{d}' if d != 14 else '') + (f' {v}' if v else '')
            + f' · {r["rung"]} {r["steps"] // 1000}k')


def ticks(lo, hi, n=8):
    step = next(s for s in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5) if (hi - lo) / s <= n)
    return [round(v * step, 4) for v in range(math.ceil(lo / step), math.floor(hi / step) + 1)]


def bars(o, fr, cols):
    """Second panel: 1_95 FPR at the pooled rotating-fold threshold, one group per frontier student (fast to slow),
    numbered as in the scatter's key."""
    num = {id(r): i + 1 for i, r in enumerate(fr)}
    items = [(r, probe_of(r['name'])) for r in fr]
    missing = sum(1 for _, c in items if not c)
    items = [(r, c['pooled']) for r, c in items if c]
    top, bot = H + 56, H + H2 - 64
    ymax = max([0.03] + [c[k] for _, c in items for k in ('fpr', 'fpr_jet', 'rotating_fpr_max')])
    ymax = math.ceil(ymax * 200) / 200
    py = lambda v: bot - v / ymax * (bot - top)
    o.append(f'<text x="{L}" y="{H + 12}" font-size="15" font-weight="bold">Jet fold 1_95: false-positive rate at the '
             f"rotating folds' threshold</text>")
    o.append(f'<text x="{L}" y="{H + 30}" fill="{INK2}">pooled fpr-0.005 threshold from the five rotating folds, applied to '
             f'1_95 (a training fold); jets = frames labelled mech_plane'
             + (f'; {missing} frontier student{"s" if missing > 1 else ""} without probe.json not shown' if missing else '')
             + '</text>')
    for v in ticks(0, ymax, 6):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v:.1%}</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(0.005):.1f}" y2="{py(0.005):.1f}" stroke="{INK2}" stroke-dasharray="4 3"/>'
             f'<text x="{W - R + 6}" y="{py(0.005) + 4:.1f}" fill="{INK2}">target 0.5%</text>')
    o.append(f'<text transform="translate(18 {(top + bot) / 2}) rotate(-90)" text-anchor="middle">1_95 false-positive rate</text>')
    if not items:
        o.append(f'<text x="{L + 20}" y="{(top + bot) / 2}">no probe.json yet: run eval_folds.py probe</text>')
        return
    gw = (W - L - R) / len(items)
    bw = min(26, gw / 3.2)
    for i, (r, c) in enumerate(items):
        cx = L + gw * (i + 0.5)
        col = cols[trunk(r['arch'])[0]]
        for dx, key, fill in ((-bw - 1, 'fpr', col), (1, 'fpr_jet', INK)):
            v = min(c[key], ymax)
            o.append(f'<rect x="{cx + dx:.1f}" y="{py(v):.1f}" width="{bw:.1f}" height="{bot - py(v):.1f}" fill="{fill}" '
                     f'fill-opacity="0.85"><title>{r["name"]} {key} {c[key]:.2%}</title></rect>'
                     f'<text x="{cx + dx + bw / 2:.1f}" y="{py(v) - 4:.1f}" font-size="9" text-anchor="middle">'
                     f'{c[key] * 100:.1f}</text>')
        o.append(f'<line x1="{cx - bw - 4:.1f}" x2="{cx + bw + 4:.1f}" y1="{py(c["rotating_fpr_max"]):.1f}" '
                 f'y2="{py(c["rotating_fpr_max"]):.1f}" stroke="{BOX}" stroke-width="3"/>')
        o.append(f'<text x="{cx:.1f}" y="{bot + 15}" font-size="11" font-weight="bold" text-anchor="middle">'
                 f'{num[id(r)]}</text><text x="{cx:.1f}" y="{bot + 28}" font-size="9" text-anchor="middle" '
                 f'fill="{INK2}">{r["x_yamnet200"]:.1f}x</text>')
    o.append(f'<rect x="{L}" y="{top}" width="{W - L - R}" height="{bot - top}" fill="none" stroke="{BOX}"/>')
    lx, ly = W - R + 20, top + 10
    for k, (fill, lab) in enumerate([('#888', '1_95, all negatives'), (INK, '1_95, jet frames only')]):
        o.append(f'<rect x="{lx}" y="{ly + k * 18}" width="12" height="12" fill="{fill}"/>'
                 f'<text x="{lx + 18}" y="{ly + 10 + k * 18}" font-size="11">{lab}</text>')
    o.append(f'<line x1="{lx}" x2="{lx + 12}" y1="{ly + 42}" y2="{ly + 42}" stroke="{BOX}" stroke-width="3"/>'
             f'<text x="{lx + 18}" y="{ly + 46}" font-size="11">worst rotating fold</text>'
             f'<text x="{lx}" y="{ly + 70}" font-size="10" fill="{INK2}">all-negatives bar in the</text>'
             f'<text x="{lx}" y="{ly + 83}" font-size="10" fill="{INK2}">student\'s trunk-width colour</text>')


def refs():
    """(label, value) for the era baseline and the teacher scored by the students' own harness (ladder_record.REFS)."""
    out = []
    for (_, p), lab in zip(LR.REFS, ('era baseline', None, 'teacher, same harness')):
        if lab and os.path.exists(p):
            out.append((lab, LR.sx(p)[0]))
    return out or [('era baseline', BASELINE)]


def main():
    rows = [r for r in load_all() if r.get('classes') == BRH]
    if not rows:
        sys.exit('no seed-1 buzz+rain+human students with a speed in the ladder yet')
    cols = width_colours(rows)
    rf = refs()
    xs, ys = [r['x_yamnet200'] for r in rows], [r['headline'] for r in rows]
    x0, x1 = min(xs + [1]) - 0.3, max(xs) + 0.4
    y0, y1 = min(ys) - 0.02, max(ys + [v for _, v in rf]) + 0.015     # a reference below every point is noted, not drawn
    px = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    py = lambda v: H - B - (v - y0) / (y1 - y0) * (H - B - T)
    fr = pareto(rows)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + H2}" font-family="sans-serif" font-size="12" fill="{INK}">',
         f'<rect width="{W}" height="{H + H2}" fill="#fff"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold">Speed / sensitivity frontier: buzz+rain+human '
         f'students, every rung, seed 1, each at its stop-rule budget</text>',
         f'<text x="{L}" y="40" fill="{INK2}">teacher {D.TEACHER} · {len(rows)} students '
         f'({sum(r["pool"] == "clean" for r in rows)} clean) · {len(fr)} non-dominated · speed = the fastest timed '
         f'export of each graph</text>']
    for v in ticks(y0, y1, 9):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for v in ticks(x0, x1, 12):
        o.append(f'<line y1="{T}" y2="{H - B}" x1="{px(v):.1f}" x2="{px(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{px(v):.1f}" y="{H - B + 18}" text-anchor="middle">{v:g}x</text>')
    o.append(f'<line x1="{px(1):.1f}" x2="{px(1):.1f}" y1="{T}" y2="{H - B}" stroke="{INK2}" stroke-dasharray="4 3"/>'
             f'<text x="{px(1) + 4:.1f}" y="{T + 14}" fill="{INK2}">YAMNet speed</text>')
    marks = [(px(r['x_yamnet200']), py(r['headline'])) for r in rows]
    for lab, v in rf:                     # reference label beside its line, in the stretch with fewest points
        t, y = f'{lab} {v:.3f}', py(v)
        w = 6.4 * len(t)
        if v < y0:
            o.append(f'<text x="{W - R - 6}" y="{H - B - 8}" text-anchor="end" fill="{INK2}">{t} (below the axis)</text>')
            continue
        crowd = lambda xe, dy: sum(xe - w - 14 < a < xe + 14 and y + dy - 22 < b < y + dy + 12 for a, b in marks)
        xe, dy = min(((xe, dy) for dy in (15, -6) for xe in range(int(W - R - 4), int(L + w), -10)),
                     key=lambda c: (crowd(*c), -c[0]))
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{INK2}" stroke-dasharray="4 3"/>'
                 f'<text x="{xe}" y="{y + dy:.1f}" text-anchor="end" fill="{INK2}" stroke="#fff" stroke-width="3" '
                 f'paint-order="stroke">{t}</text>')
    o.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - B - T}" fill="none" stroke="{BOX}"/>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle">speed, x YAMNet (GPU, 200 s audio; higher is faster)</text>')
    o.append(f'<text transform="translate(18 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">'
             f'sensitivity (excl. quiet) @ 0.5% FPR</text>')
    by = {}                               # one student across rungs: a thin line from its smaller rung to its larger
    for r in rows:
        by.setdefault(ident(r)[:-1], []).append(r)
    for rs in by.values():
        if len(rs) > 1:
            rs.sort(key=lambda r: r['rung'])
            o.append(f'<polyline fill="none" stroke="#b0b0b0" stroke-width="1.2" points="'
                     + ' '.join(f'{px(r["x_yamnet200"]):.1f},{py(r["headline"]):.1f}' for r in rs) + '"/>')
    o.append(f'<polyline fill="none" stroke="{INK}" stroke-width="1.5" stroke-dasharray="2 3" points="'
             + ' '.join(f'{px(r["x_yamnet200"]):.1f},{py(r["headline"]):.1f}' for r in fr) + '"/>')
    for r in sorted(rows, key=lambda r: (r in fr, r['pool'] == 'clean', r['rung'])):   # frontier and clean on top
        x, y = px(r['x_yamnet200']), py(r['headline'])
        speed = f'{r["x_yamnet200"]:.2f}x ({r.get("export", "stft")}' + (
            f', timed as {r["speed_from"]}' if r.get('speed_from') not in (None, r['name']) else '') + ')'
        o.append(f'<g>{marker(family(r["frontend"])[2], x, y, cols[trunk(r["arch"])[0]], r["rung"], faded=r["pool"] != "clean", dashed=pre_mmt(r))}'
                 f'<title>{r["name"]} ({r["pool"]})\nheadline {r["headline"]:.3f}  {speed}  '
                 f'{r["steps"] // 1000}k steps</title></g>')
    # frontier callouts: a number beside each point (right, left, then further up/down, clear of markers and
    # other numbers); the key under the legend names them
    boxes = [(a - 8, b - 8, a + 8, b + 8) for a, b in marks]
    clear = lambda bx: not any(bx[0] < c and bx[2] > a and bx[1] < d and bx[3] > b for a, b, c, d in boxes)
    for i, r in enumerate(fr):
        x, y, t = px(r['x_yamnet200']), py(r['headline']), str(i + 1)
        w = 7 * len(t) + 2
        cands = [(dx, dy) for dy in (0, -13, 13, -24, 24) for dx in (10, -10 - w, -w / 2)
                 if not (dy == 0 and dx == -w / 2)]
        dx, dy = next((c for c in cands if clear((x + c[0], y + c[1] - 7, x + c[0] + w, y + c[1] + 6))), cands[0])
        lx, ly = x + dx, y + dy + 5
        boxes.append((lx, ly - 12, lx + w, ly + 1))
        o.append(f'<text x="{lx + 1:.1f}" y="{ly:.1f}" font-size="12" font-weight="bold" stroke="#fff" stroke-width="3" '
                 f'paint-order="stroke">{t}</text>')
    # legend, only what is drawn
    lx, ly = W - R + 20, T + 4
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Trunk width (fill)</text>')
    ws = sorted({trunk(r['arch'])[0] for r in rows}, key=float)
    for i, w in enumerate(ws):
        o.append(f'<rect x="{lx}" y="{ly + 8 + i * 18}" width="12" height="12" fill="{cols[w]}"/>'
                 f'<text x="{lx + 18}" y="{ly + 18 + i * 18}">a{w}</text>')
    ly += 18 * len(ws) + 30
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Front end (shape)</text>')
    fams = [f for f in FAMILIES if any(family(r['frontend']) == f for r in rows)]
    for i, (name, _, kind) in enumerate(fams):
        o.append(marker(kind, lx + 6, ly + 16 + i * 19, '#888')
                 + f'<text x="{lx + 22}" y="{ly + 20 + i * 19}">{name}</text>')
    ly += 19 * len(fams) + 30
    o.append(f'<text x="{lx}" y="{ly}" font-weight="bold">Rung (outline)</text>')
    rungs = sorted({r['rung'] for r in rows})
    for i, k in enumerate(rungs):
        o.append(marker('circle', lx + 6, ly + 16 + i * 21, '#bbb', k)
                 + f'<text x="{lx + 22}" y="{ly + 20 + i * 21}">{RUNG_NAME.get(k, k)}</text>')
    ly += 21 * len(rungs) + 14
    extra = [(marker('circle', lx + 6, ly + 4, '#bbb', faded=True), 'quarantined pool (contaminated)')] \
        if any(r['pool'] != 'clean' for r in rows) else []
    if any(pre_mmt(r) for r in rows):
        extra.append((marker('circle', lx + 6, ly + 4 + 21 * len(extra), '#bbb', dashed=True),
                      'speed timed before mmt (slow)'))
    for i, (m, t) in enumerate(extra):
        o.append(m + f'<text x="{lx + 22}" y="{ly + 8 + i * 21}" font-size="11">{t}</text>')
    ly += 21 * len(extra) + 8
    o.append(f'<line x1="{lx}" x2="{lx + 14}" y1="{ly}" y2="{ly}" stroke="#b0b0b0" stroke-width="1.2"/>'
             f'<text x="{lx + 22}" y="{ly + 4}" font-size="11">same student, across rungs</text>'
             f'<line x1="{lx}" x2="{lx + 14}" y1="{ly + 20}" y2="{ly + 20}" stroke="{INK}" stroke-dasharray="2 3"/>'
             f'<text x="{lx + 22}" y="{ly + 24}" font-size="11">non-dominated</text>'
             f'<text x="{lx}" y="{ly + 50}" font-weight="bold">Non-dominated, fast to slow</text>')
    for i, r in enumerate(fr):
        yy = ly + 70 + i * 26
        o.append(f'<text x="{lx}" y="{yy}" font-size="11" font-weight="bold">{i + 1}</text>'
                 f'<text x="{lx + 16}" y="{yy}" font-size="11">{label(r)}</text>'
                 f'<text x="{lx + 16}" y="{yy + 12}" font-size="10" fill="{INK2}">{r["headline"]:.3f} at '
                 f'{r["x_yamnet200"]:.2f}x{" (timed pre-mmt)" if pre_mmt(r) else ""}</text>')
    bars(o, fr, cols)
    o.append('</svg>')
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontier.svg')
    open(out, 'w').write('\n'.join(o))
    print(out, len(rows), 'students,', len(fr), 'on the frontier,',
          sum(r['pool'] == 'clean' for r in rows), 'clean,', sum(pre_mmt(r) for r in rows), 'timed before mmt')


if __name__ == '__main__':
    main()
