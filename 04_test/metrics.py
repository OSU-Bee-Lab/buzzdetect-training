import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

FRAMELENGTH = 0.96


def read_results(dir_results):
    paths = glob.glob(os.path.join(dir_results, '**', '*_buzzdetect.csv'), recursive=True)
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        ident = re.sub(r'_buzzdetect\.csv$', '', os.path.relpath(path, dir_results))
        df['ident'] = ident
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True).rename(columns={'start': 'start_frame'})
    if 'ins_buzz' in result.columns:
        result = result.rename(columns={'ins_buzz': 'activation_ins_buzz'})
    return result


def join_results(dir_results, annotations, framelength=FRAMELENGTH):
    overlap_s = 0.1 * framelength

    results = read_results(dir_results)
    results['end_frame'] = results['start_frame'] + framelength

    ann = annotations.rename(columns={'ident': 'ident_annotation'}).copy()
    ann['end_annotation_overlap'] = ann['end_annotation'] - overlap_s
    ann['start_annotation_overlap'] = ann['start_annotation'] + overlap_s

    merged = results.merge(ann, left_on='ident', right_on='ident_annotation')
    overlap = merged[
        (merged['end_annotation_overlap'] > merged['start_frame']) &
        (merged['start_annotation_overlap'] <= merged['end_frame'])
    ].copy()

    def collapse(s):
        vals = sorted(v for v in s.dropna().unique())
        return '; '.join(str(v) for v in vals) if vals else np.nan

    frame_events = overlap.groupby(['ident', 'start_frame']).agg(
        label=('label', collapse),
        buzz_pitch=('buzz_pitch', collapse),
        buzz_volume=('buzz_volume', collapse),
        multiple=('label', lambda x: len(x) > 1),
    ).reset_index()

    results_join = (
        results[['ident', 'start_frame', 'activation_ins_buzz']]
        .merge(frame_events, on=['ident', 'start_frame'], how='left')
    )
    results_join['label'] = results_join['label'].fillna('ambient_background')
    results_join['correct'] = results_join['label'].str.contains('ins_buzz', na=False)

    return results_join


def metrics_by_group(results_join, groups=None):
    if groups is None:
        groups = []

    group_cols = groups + ['activation_ins_buzz']

    agg = results_join.groupby(group_cols).agg(
        frames_here=('correct', 'count'),
        buzzes_here=('correct', 'sum'),
    ).reset_index()
    agg['nonbuzzes_here'] = agg['frames_here'] - agg['buzzes_here']
    agg = agg.sort_values(groups + ['activation_ins_buzz'], ascending=[True] * len(groups) + [False])

    def cumstats(g):
        g = g.copy()
        g['frames_above'] = g['frames_here'].cumsum()
        g['buzzes_above'] = g['buzzes_here'].cumsum()
        g['nonbuzzes_above'] = g['nonbuzzes_here'].cumsum()
        total_buzzes = g['buzzes_here'].sum()
        total_nonbuzzes = g['nonbuzzes_here'].sum()
        g['precision'] = g['buzzes_above'] / g['frames_above']
        g['sensitivity'] = g['buzzes_above'] / total_buzzes
        g['fpr'] = g['nonbuzzes_above'] / total_nonbuzzes
        return g

    result = agg.groupby(groups, group_keys=False).apply(cumstats) if groups else cumstats(agg)

    return result.rename(columns={'activation_ins_buzz': 'threshold'})[
        groups + ['threshold', 'precision', 'sensitivity', 'fpr']
    ]


def compute_model_metrics(modelname, annotations, framelength=FRAMELENGTH):
    dir_results = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, 'results')
    results_join = join_results(dir_results, annotations, framelength)
    return metrics_by_group(results_join)


if __name__ == '__main__':
    annotations = pd.read_csv(os.path.join(cfg.TEST_DIR, 'annotations.csv'))

    models_tested = [
        d for d in os.listdir(cfg.DIR_MODELS)
        if os.path.isdir(os.path.join(cfg.DIR_MODELS, d, cfg.SUBDIR_TESTS, 'results'))
    ]

    all_metrics = []
    for modelname in models_tested:
        print(f'Computing metrics for {modelname}')
        m = compute_model_metrics(modelname, annotations)
        m.insert(0, 'model', modelname)
        path_out = os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_TESTS, cfg.FNAME_METRICS)
        m.drop(columns='model').to_csv(path_out, index=False)
        print(f'  saved {path_out}')
        all_metrics.append(m)

    if not all_metrics:
        print('No tested models found.')
    else:
        df = pd.concat(all_metrics, ignore_index=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        for model, group in df.groupby('model'):
            ax.plot(group['precision'], group['sensitivity'], label=model)
        ax.set_xlabel('Precision')
        ax.set_ylabel('Sensitivity')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.legend()
        ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.show()
