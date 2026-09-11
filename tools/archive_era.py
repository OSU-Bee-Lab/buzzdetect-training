"""Close out an experiment era — LOOP.md's cutover procedure, mechanised.

An era ends when something makes its logged numbers stop meaning what they said:
the metric, the eval design, the fold roster, or the training data. At that point
`log.jsonl` gets archived and started fresh. Doing that by hand is how the
2026-08 archive ended up in a gitignored directory and evaporated.

    python tools/archive_era.py --slug cv-medium-v2            # preflight only
    python tools/archive_era.py --slug cv-medium-v2 --write    # do it

Preview by default, same as log_entry.py. --write refuses unless preflight is
clean.

WHAT THIS DOES NOT DO
---------------------
It writes the era README with every derivable table filled in and the prose left
as TODO markers. The prose is the point of the archive — what ended the era,
which verdicts survive as leads, what a later reader would misread. Fill it in
before committing; the script will not invent it.

It also does not rewrite LOOP.md's Baseline section, prune worktrees, or commit.
Those need judgment or are destructive; it prints them as next steps.

No TensorFlow, no pandas.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, 'archive')

# Set products worth snapshotting: all gitignored, all required to know what a
# logged float measured.
SET_FILES = ('annotations.csv', 'folds.csv', 'summary_per_class.csv',
             'summary_per_fold.csv', 'config_extract.json')

# setup_worktree.sh replaces checked-out dirs with symlinks, so every worktree
# reads as dirty on these. Not real changes; don't block on them.
SYMLINK_NOISE = re.compile(
    r'^(embedders/|audio$|02_set/sets/[^/]+/(embeddings|audio|folds\.csv'
    r'|annotations\.csv|summary_))')


def git(*args, cwd=ROOT, check=True):
    r = subprocess.run(('git',) + args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def git_ok(*args, cwd=ROOT):
    return subprocess.run(('git',) + args, cwd=cwd,
                          capture_output=True).returncode == 0


# ---------------------------------------------------------------- discovery

def read_log(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def orphan_notes():
    """notes.md from commits with no surviving branch, keyed by experiment name.

    Two sources, and both matter. `git fsck` finds commits whose branch was
    deleted and which nothing has pinned yet — that is how 34 experiments' notes
    were recovered in 2026-09, and the next `git gc` would have taken them.
    `refs/archive/*` holds the ones a previous run of this script already
    rescued: they are no longer dangling precisely *because* they were pinned,
    so fsck alone would report them missing on the next cutover.
    """
    out = {}

    for ref in git('for-each-ref', '--format=%(refname)',
                   'refs/archive/').splitlines():
        body = notes_on(ref)
        if body is not None:
            out[ref.rsplit('/', 1)[-1]] = (ref, body)

    fsck = subprocess.run(('git', 'fsck', '--lost-found'), cwd=ROOT,
                          capture_output=True, text=True).stdout
    for line in fsck.splitlines():
        if 'dangling commit' not in line:
            continue
        sha = line.split()[-1]
        body = subprocess.run(('git', 'show', f'{sha}:notes.md'), cwd=ROOT,
                              capture_output=True, text=True)
        if body.returncode:
            continue
        first = body.stdout.splitlines()[0] if body.stdout else ''
        name = re.sub(r'[^a-zA-Z0-9_-]', '', first.lstrip('# ').strip())
        if name:
            out.setdefault(name, (sha, body.stdout))
    return out


def live_branches():
    names = git('branch', '--format=%(refname:short)').splitlines()
    return [n for n in names if n.startswith('exp/')]


def notes_on(ref):
    r = subprocess.run(('git', 'show', f'{ref}:notes.md'), cwd=ROOT,
                       capture_output=True, text=True)
    return None if r.returncode else r.stdout


def worktree_dirt():
    """Uncommitted tracked changes per worktree, symlink noise filtered out."""
    dirty = {}
    for line in git('worktree', 'list', '--porcelain').splitlines():
        if not line.startswith('worktree '):
            continue
        wt = line.split(' ', 1)[1]
        if os.path.realpath(wt) == os.path.realpath(ROOT):
            continue
        status = subprocess.run(('git', 'status', '--porcelain'), cwd=wt,
                                capture_output=True, text=True).stdout
        real = [l for l in status.splitlines()
                if l and not l.startswith('?? ')
                and not SYMLINK_NOISE.match(l[3:])]
        if real:
            dirty[os.path.basename(wt)] = real
    return dirty


# ---------------------------------------------------------------- preflight

def preflight(entries, setname):
    """Returns (blockers, warnings, facts). --write requires no blockers."""
    blockers, warnings, facts = [], [], {}

    if not entries:
        blockers.append('log.jsonl is empty — nothing to archive.')
        return blockers, warnings, facts

    dates = sorted(e['date'] for e in entries if e.get('date'))
    facts['first_date'] = dates[0] if dates else None
    facts['last_date'] = dates[-1] if dates else None
    facts['n'] = len(entries)

    if not facts['first_date']:
        blockers.append('No entry carries a date; cannot name the era directory.')

    # 1. Every logged branch must still be reachable, somewhere.
    live = set(live_branches())
    dang = orphan_notes()
    missing, recovered = [], []
    for e in entries:
        br = e.get('branch', '')
        if not br or br == 'main':
            continue
        slug = br[len('exp/'):] if br.startswith('exp/') else br
        if br in live and notes_on(br) is not None:
            continue
        if slug in dang:
            recovered.append(slug)
        else:
            missing.append(f"{e['name']} ({br})")
    facts['recovered'] = sorted(set(recovered))
    if missing:
        warnings.append(
            'No notes.md found for: ' + ', '.join(sorted(set(missing)))
            + '. Their log entries are all that will survive.')

    # 2. Unpushed branches. This is the one that actually loses work — the
    #    worktree gets pruned, the disk dies, and the branch was never anywhere
    #    else.
    unpushed = []
    for br in live:
        if not git_ok('rev-parse', '--verify', f'origin/{br}'):
            unpushed.append(f'{br} (no remote branch)')
        elif git('rev-parse', br) != git('rev-parse', f'origin/{br}'):
            unpushed.append(f'{br} (ahead of origin)')
    if unpushed:
        blockers.append(
            'Branches not pushed:\n      ' + '\n      '.join(sorted(unpushed))
            + '\n    Push them first — archiving is what makes worktrees'
              ' disposable, and the branch is the only durable copy.')

    # 3. Uncommitted work in worktrees. Pruning destroys it, and the notes
    #    harvested below come from the *committed* tree, so a rewritten
    #    notes.md would be archived stale.
    dirt = worktree_dirt()
    if dirt:
        lines = []
        for wt, changes in sorted(dirt.items()):
            lines.append(f'{wt}: ' + ', '.join(c[3:] for c in changes[:6]))
        blockers.append(
            'Uncommitted tracked changes in worktrees:\n      '
            + '\n      '.join(lines)
            + '\n    Commit or discard them. A rewritten notes.md here would be'
              ' archived in its stale committed form.')

    # 4. Set snapshot must exist.
    setdir = os.path.join(ROOT, '02_set', 'sets', setname)
    absent = [f for f in SET_FILES if not os.path.exists(os.path.join(setdir, f))]
    if absent:
        blockers.append(f"Set '{setname}' is missing {', '.join(absent)}."
                        ' Run build.R — the snapshot is what makes the log'
                        ' interpretable.')
    facts['setdir'] = setdir
    return blockers, warnings, facts


# ---------------------------------------------------------------- derivation

def fold_roster(setdir, translation='general'):
    """Per rotating fold: annotation counts and ins_buzz load."""
    tpath = os.path.join(setdir, 'translations', f'{translation}.csv')
    if not os.path.exists(tpath):
        return None
    with open(tpath) as f:
        tr = {r['from']: r['to'] for r in csv.DictReader(f)}
    with open(os.path.join(setdir, 'folds.csv')) as f:
        folds = {r['ident']: r for r in csv.DictReader(f)}
    rot = sorted({r['fold'] for r in folds.values() if r['role'] == 'rotate'})
    an, bn, bs, ts = Counter(), Counter(), Counter(), Counter()
    span = {}
    with open(os.path.join(setdir, 'annotations.csv')) as f:
        for r in csv.DictReader(f):
            fold = folds.get(r['ident'], {}).get('fold')
            if fold not in rot:
                continue
            d = float(r['duration'])
            an[fold] += 1
            ts[fold] += d
            span[fold] = max(span.get(fold, 0.0), float(r['end']))
            if tr.get(r['label']) == 'ins_buzz':
                bn[fold] += 1
                bs[fold] += d
    rows = [(f, an[f], bn[f], bs[f], ts[f], span.get(f, 0.0) / 3600) for f in rot]
    return rows


def roster_table(rows):
    out = ['| rotating fold (deployment) | annots | buzz | buzz s | annot s | span h |',
           '|---|---|---|---|---|---|']
    for f, a, b, bsec, tsec, hrs in rows:
        out.append(f'| {f} | {a} | {b} | {bsec:.0f} | {tsec:.0f} | {hrs:.1f} |')
    tot = [sum(c) for c in zip(*[(r[1], r[2], r[3], r[4], r[5]) for r in rows])]
    out.append(f'| **total** | **{tot[0]}** | **{tot[1]}** | **{tot[2]:.0f}** '
               f'| **{tot[3]:.0f}** | **{tot[4]:.1f}** |')
    return '\n'.join(out)


def top_results(entries, k=5):
    scored = []
    for e in entries:
        m = (e.get('metrics') or {})
        v = m.get('sens_at_fpr0.005_persite')
        if isinstance(v, (int, float)):
            scored.append((v, e['name'], e.get('trust', '?')))
    scored.sort(reverse=True)
    return scored[:k]


# ---------------------------------------------------------------- README

README_TEMPLATE = """# Era: {slug}

**{n} runs, {first} → {last}.**
<!-- TODO: one line on what this era was. -->

Closed on {last}<!-- TODO: by what? the change that made these numbers stop
meaning what they said. -->

## What made these comparable

| | |
|---|---|
| Set | `{setname}` — {extract} |
| Folds | {nfolds} rotating (`role: rotate`), {ntrain} train-only |
| Metric | `total` row `sensitivity` at `fpr` 0.005 in `folds_sx.csv` — plain mean over folds, threshold set per fold on its own held-out audio |
| Translation | <!-- TODO --> |
| Embedder | <!-- TODO --> |

## Baseline and best

<!-- TODO: name the baseline entry and its number. -->

{top}

<!-- TODO: flag any entry whose float is not what it looks like — incomplete
runs, artifacts. The `trust` field is the first thing to read. -->

## Gotchas

<!-- TODO: this is the section that earns the archive. What would a reader a
year from now get wrong? Translation changes mid-era, annotations moving under
the results, folds too thin to read, known eval bugs. -->

## Fold roster at cutover

Counts from `set/`, `ins_buzz` under the translation named above.

{roster}

## Notes and code

`notes/` holds {nnotes}.
{recovered}
Branches: `exp/*` on origin where they survive.

## Reading these forward

<!-- TODO: which verdicts are worth rerunning, which are dead ends. Positive
results rarely survive a data change; dead ends usually do. -->
"""


def build_readme(entries, facts, setname, slug, roster, nnotes, recovered):
    setdir = facts['setdir']
    with open(os.path.join(setdir, 'config_extract.json')) as f:
        cfg = json.load(f)
    extract = ', '.join(f'`{k}: {v}`' for k, v in cfg.items() if k != 'setname')
    with open(os.path.join(setdir, 'folds.csv')) as f:
        roles = Counter(r['role'] for r in csv.DictReader(f))
    top = '\n'.join(f'| `{n}` | {v} | {t} |' for v, n, t in top_results(entries))
    if top:
        top = '| Result | | Trust |\n|---|---|---|\n' + top
    rec = ''
    if recovered:
        rec = ('\n**{} recovered from dangling commits** (their branches were '
               'already deleted): {}. Code at `refs/archive/<name>`.\n'.format(
                   len(recovered), ', '.join(f'`{r}`' for r in recovered)))
    return README_TEMPLATE.format(
        slug=slug, n=facts['n'], first=facts['first_date'],
        last=facts['last_date'], setname=setname, extract=extract,
        nfolds=roles.get('rotate', 0), ntrain=roles.get('train', 0),
        top=top, roster=roster or '<!-- roster unavailable -->',
        nnotes=nnotes, recovered=rec)


# ---------------------------------------------------------------- write

def do_write(entries, facts, setname, slug, translation, tag):
    era = f"{facts['first_date']}_{slug}"
    dest = os.path.join(ARCHIVE, era)
    if os.path.exists(dest):
        sys.exit(f'error: {dest} already exists.')
    os.makedirs(os.path.join(dest, 'notes'))
    os.makedirs(os.path.join(dest, 'set', 'translations'))

    shutil.copy(os.path.join(ROOT, 'log.jsonl'), os.path.join(dest, 'log.jsonl'))

    setdir = facts['setdir']
    for f in SET_FILES:
        shutil.copy(os.path.join(setdir, f), os.path.join(dest, 'set', f))
    tdir = os.path.join(setdir, 'translations')
    for f in sorted(os.listdir(tdir)):
        if f.endswith('.csv'):
            shutil.copy(os.path.join(tdir, f),
                        os.path.join(dest, 'set', 'translations', f))

    # Notes: live branches first, then orphans recovered from dangling commits.
    written = set()
    for br in live_branches():
        body = notes_on(br)
        if body is None:
            continue
        name = br[len('exp/'):]
        with open(os.path.join(dest, 'notes', f'{name}.md'), 'w') as f:
            f.write(body)
        written.add(name)

    recovered = []
    for name, (sha, body) in sorted(orphan_notes().items()):
        if name in written:
            continue
        with open(os.path.join(dest, 'notes', f'{name}.md'), 'w') as f:
            f.write(body)
        if not sha.startswith('refs/'):
            git('update-ref', f'refs/archive/{name}', sha)   # survive gc
        written.add(name)
        recovered.append(name)

    roster = fold_roster(setdir, translation)
    with open(os.path.join(dest, 'README.md'), 'w') as f:
        f.write(build_readme(entries, facts, setname, slug,
                             roster_table(roster) if roster else None,
                             len(written), recovered))

    # Index row.
    index = os.path.join(ARCHIVE, 'README.md')
    if os.path.exists(index):
        with open(index, 'a') as f:
            f.write(f'\n<!-- added by archive_era.py -->\n'
                    f'| [`{era}`]({era}/) | {facts["n"]} | '
                    f'{facts["first_date"]} → {facts["last_date"]} | '
                    f'TODO metric | TODO ended by |\n')

    open(os.path.join(ROOT, 'log.jsonl'), 'w').close()
    if tag:
        git('tag', '-a', tag, '-m', f'Final state of the {era} log.')
    return dest, era, recovered


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--slug', required=True,
                   help='era name, e.g. cv-medium-v2. Directory becomes '
                        '<first-entry-date>_<slug>.')
    p.add_argument('--set', default='medium', dest='setname')
    p.add_argument('--translation', default='general',
                   help='translation the roster table counts ins_buzz under')
    p.add_argument('--tag', default=None,
                   help='annotated tag for the pre-truncation commit')
    p.add_argument('--write', action='store_true',
                   help='actually archive; refuses if preflight has blockers')
    a = p.parse_args()

    entries = read_log(os.path.join(ROOT, 'log.jsonl'))
    blockers, warnings, facts = preflight(entries, a.setname)

    print(f'era      : {facts.get("first_date")}_{a.slug}')
    print(f'runs     : {facts.get("n", 0)}'
          f'  ({facts.get("first_date")} → {facts.get("last_date")})')
    print(f'set      : {a.setname}')
    for w in warnings:
        print(f'\n  warn: {w}')
    for b in blockers:
        print(f'\n  BLOCK: {b}')

    if blockers:
        print('\nRefusing. Fix the blockers above.')
        sys.exit(1)
    if not a.write:
        print('\nPreflight clean. Re-run with --write to archive.')
        return

    dest, era, recovered = do_write(entries, facts, a.setname, a.slug,
                                    a.translation, a.tag)
    print(f'\nwrote {dest}')
    print(f'  orphaned notes rescued: {len(recovered)}')
    print('  log.jsonl truncated')
    print(f"""
Not done for you — these need judgment:

  1. Fill the TODO markers in {os.path.relpath(dest, ROOT)}/README.md and the
     new row in archive/README.md. The prose is the point of the archive.
  2. Rewrite LOOP.md's Baseline section — there is no baseline until one is
     rerun on the new data, and the old model is no longer a valid comparator.
  3. Do NOT prune the worktrees. LOOP.md's step 6 is explicit about why:
     a worktree that only trained is ~50 MB, and removing them has
     repeatedly destroyed the only copy of something (an embedder a
     gitignore gap never captured; thirty-one branches' notes.md).
     The branch is the durable record; the worktree is a free second copy.
  4. Commit, and push main plus 'refs/archive/*:refs/archive/*'.
""")


if __name__ == '__main__':
    main()
