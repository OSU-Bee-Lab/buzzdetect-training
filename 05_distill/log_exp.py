"""Stamp a loop experiment's slug on its students' rows in 05_distill/log.jsonl: LOOP.md step 5, distillation arm.

    python 05_distill/log_exp.py --exp <slug> --runs <name> [<name> ...] [--check]

main.py's record stage writes one row per judged student; this adds `"exp": "<slug>"` to the current teacher's
rows with those names, which is what retires the experiment's HANDOFF.md (tools/human/agent_loop.sh) and what
tools/finish_experiment.sh --arm distill commits. --check only validates: every name has exactly one row, and
none is stamped for another experiment. Standard library only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpaths as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--check', action='store_true', help='validate only, write nothing')
    a = ap.parse_args()
    rows = D.read_log(teacher=None)
    mine = {i: r for i, r in enumerate(rows) if r.get('teacher') == D.TEACHER and r.get('name') in a.runs}
    errors = []
    for n in a.runs:
        hits = [r for r in mine.values() if r['name'] == n]
        if len(hits) != 1:
            errors.append(f'{n}: {len(hits)} rows for teacher {D.TEACHER} in {D.LOG} (want 1; has main.py recorded it?)')
        elif hits[0].get('exp', a.exp) != a.exp:
            errors.append(f'{n}: already stamped for experiment {hits[0]["exp"]}')
    if errors:
        sys.exit('log_exp.py: ' + '\n  '.join(errors))
    if a.check:
        return
    for i in mine:
        rows[i]['exp'] = a.exp
    D.write_log(rows)
    print(f'[log_exp] {len(mine)} rows stamped exp={a.exp}')


if __name__ == '__main__':
    main()
