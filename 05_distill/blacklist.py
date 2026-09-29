"""Derive the deployment blacklist for student distillation audio.

Identity rule: for an audio file  <project>/.../<deployment>/<recorder>/file
the *deployment* is dirname(dirname(file)) (the "grandparent" dir). A file
whose deployment equals a blacklisted deployment is excluded. The teacher's
training folds (folds.csv 'fold' column / config_model.json 'folds_train') are
recorder dirs relative to the audio root, so blacklist = dirname(fold). A fold
of the form <project>/<recorder> therefore blacklists the whole project; a
file directly under a project has the project as its deployment.

Read-only on the audio tree. No GPU. Run: python 05_distill/blacklist.py
"""
import argparse, collections, csv, json, os

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.local-scratch-not-committed')
AUDIO_EXT = {'.wav', '.mp3', '.wma', '.flac', '.m4a', '.ogg', '.mp4', '.mts'}
# Not deployment audio: public datasets, trash, weather data. Reported, never used.
NOT_DEPLOYMENT = {'[trash]', 'weather station data', 'Luke - External Data Sources'}


def deployment(rel):
    d = os.path.dirname(os.path.dirname(rel))
    return d if d else rel.split('/')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-root', default='/media/server storage/experiments')
    ap.add_argument('--model-dir', default='/home/luke/projects/buzzdetect-training/models/v4-ft-ps-e60-moderate')
    ap.add_argument('--out', default=os.path.join(SCRATCH, 'blacklist.txt'))
    a = ap.parse_args()

    folds = set(json.load(open(os.path.join(a.model_dir, 'config_model.json')))['folds_train'])
    folds |= {r['fold'] for r in csv.DictReader(open(os.path.join(a.model_dir, 'folds.csv')))}
    black = sorted({os.path.dirname(f) or f for f in folds})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write('\n'.join(black) + '\n')
    bset = set(black)

    files = collections.Counter()      # deployment -> n files
    sizes = collections.Counter()
    for dp, dn, fn in os.walk(a.audio_root):
        rel_dir = os.path.relpath(dp, a.audio_root)
        if rel_dir.split('/')[0] in NOT_DEPLOYMENT:
            dn[:] = []
            continue
        for f in fn:
            if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                rel = f if rel_dir == '.' else rel_dir + '/' + f
                dep = deployment(rel)
                files[dep] += 1
                sizes[dep] += os.path.getsize(os.path.join(dp, f))

    keep = [d for d in files if d not in bset]
    hit = [d for d in files if d in bset]
    unmatched = sorted(bset - set(files))
    print(f'folds {len(folds)} -> blacklisted deployment dirs {len(black)} (unmatched in tree: {unmatched})')
    print(f'deployments total {len(files)}, blacklisted {len(hit)} ({sum(files[d] for d in hit)} files, {sum(sizes[d] for d in hit)/1e9:.0f} GB)')
    print(f'remaining deployments {len(keep)}, files {sum(files[d] for d in keep)}, {sum(sizes[d] for d in keep)/1e9:.0f} GB')
    per = collections.defaultdict(lambda: [0, 0, 0])
    for d in keep:
        p = per[d.split('/')[0]]
        p[0] += 1; p[1] += files[d]; p[2] += sizes[d]
    for k, v in sorted(per.items(), key=lambda x: -x[1][2]):
        print(f'  {v[0]:5d} deps {v[1]:7d} files {v[2]/1e9:6.0f} GB  {k}')
    print('hours = GB / bitrate: mp3/wma projects are ~48 kbps (~21.6 GB per 1000 h... i.e. 1 GB ~ 46 h)')


if __name__ == '__main__':
    main()
