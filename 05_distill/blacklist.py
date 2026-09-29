"""Deployment blacklist for student distillation audio (see DESIGN.md).

Identity rule: for an audio file <project>/.../<deployment>/<recorder>/file the
*deployment* is dirname(dirname(file)); a file directly under a project counts
as its own project. The teacher's training folds are recorder dirs, so each
fold blacklists dirname(fold). Exclusion is by path prefix (a blacklisted dir
excludes everything below it). Stricter reading for `Reed - Illinois Soybean`:
its folds blacklist the whole date dir (`Reed.../2026-07-17`), not just the
recorder (`.../6_76`).

Library: `blacklist_dirs()`, `excluded(rel, black)`, `deployment(rel)`,
`walk_audio(root)`. CLI (`python 05_distill/blacklist.py`) prints a survey and
writes the list. Read-only on the audio tree. No GPU.
"""
import argparse, collections, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

AUDIO_EXT = {'.wav', '.mp3', '.wma', '.flac', '.m4a', '.ogg', '.mp4', '.mts'}
# Not deployment audio: public datasets, trash, weather data. Never used.
NOT_DEPLOYMENT = {'[trash]', 'weather station data', 'Luke - External Data Sources'}
STRICT_PROJECTS = {'Reed - Illinois Soybean'}   # blacklist the date dir (2 components)
TEACHER = 'v4-ft-ps-e60-moderate'


def teacher_model_dir(name=TEACHER):
    """models/<name>, falling back to the main checkout when run from a worktree."""
    d = os.path.join(config.DIR_MODELS, name)
    if os.path.isfile(os.path.join(d, 'config_model.json')):
        return d
    main = ROOT.split(os.sep + '.claude' + os.sep + 'worktrees' + os.sep)[0]
    return os.path.join(main, 'models', name)


def deployment(rel):
    d = os.path.dirname(os.path.dirname(rel))
    return d if d else rel.split('/')[0]


def blacklist_dirs(model_dir=None):
    model_dir = model_dir or teacher_model_dir()
    folds = set(json.load(open(os.path.join(model_dir, 'config_model.json')))['folds_train'])
    folds |= {r['fold'] for r in csv.DictReader(open(os.path.join(model_dir, 'folds.csv')))}
    out = set()
    for f in folds:
        d = os.path.dirname(f) or f
        parts = d.split('/')
        if parts[0] in STRICT_PROJECTS and len(parts) > 2:
            d = '/'.join(parts[:2])
        out.add(d)
    return sorted(out)


def excluded(rel, black):
    """True if rel (path relative to the audio root) is under a blacklisted dir
    or a non-deployment top-level dir."""
    if rel.split('/')[0] in NOT_DEPLOYMENT:
        return True
    p = rel
    while '/' in p:
        p = p.rsplit('/', 1)[0]
        if p in black:
            return True
    return False


def walk_audio(root):
    """Yield (rel, size) for every audio file under root outside NOT_DEPLOYMENT."""
    for top in sorted(os.listdir(root)):
        if top in NOT_DEPLOYMENT or not os.path.isdir(os.path.join(root, top)):
            continue
        for dp, dn, fn in os.walk(os.path.join(root, top)):
            dn.sort()
            for f in sorted(fn):
                if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                    full = os.path.join(dp, f)
                    yield os.path.relpath(full, root), os.path.getsize(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-root', default=config.AUDIO_ROOT)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--out', default=None, help='write the blacklist here')
    a = ap.parse_args()
    black = blacklist_dirs(a.model_dir)
    if a.out:
        open(a.out, 'w').write('\n'.join(black) + '\n')
    bset = set(black)
    files, sizes = collections.Counter(), collections.Counter()
    for rel, sz in walk_audio(a.audio_root):
        dep = deployment(rel)
        files[dep] += 1; sizes[dep] += sz
    hit = [d for d in files if excluded(d + '/x', bset)]
    keep = [d for d in files if d not in set(hit)]
    print(f'blacklist dirs {len(black)}; deployments total {len(files)}, blacklisted {len(hit)}, remaining {len(keep)}')
    print(f'remaining files {sum(files[d] for d in keep)}, {sum(sizes[d] for d in keep)/1e9:.0f} GB')


if __name__ == '__main__':
    main()
