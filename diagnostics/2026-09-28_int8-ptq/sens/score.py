import tensorflow as tf
import sys, os, csv, numpy as np, pandas as pd
ROOT='/home/luke/projects/buzzdetect-training/'
sys.path[:0]=[ROOT, ROOT+'03_train']
from sx import _fold_sens
W=sys.argv[1]; variants=sys.argv[2:]
MODEL=ROOT+'models/v4-ft-ps-e60-moderate/'
rot=[r['fold'] for r in csv.DictReader(open(ROOT+'02_set/sets/moderate/folds.csv')) if r['role']=='rotate']
preds=[pd.read_csv(MODEL+'folds/'+f+'/predictions.csv') for f in rot]
CEN=-1.076
def sens(vals_by_fold, col):
    rows=[]
    for p,a in zip(preds,vals_by_fold):
        d=p.copy(); d['activation_ins_buzz']=a
        cols,_=_fold_sens(d,[0.005]); rows.append(cols.loc[0.005,col])
    return rows
def load(v): return [np.load(f'{W}/act_{v}_{i}.npy') for i in range(5)]
print('alignment: head(embeddings) vs fp32 graph (graph has centre subtracted)')
hl=[np.load(f'{W}/headlogit_{i}.npy') for i in range(5)]; f32=load('fp32')
for i in range(5): print(' fold',i,'mean|d|',float(np.abs(hl[i]-CEN-f32[i]).mean()).__round__(4), 'max',float(np.abs(hl[i]-CEN-f32[i]).max()).__round__(3))
res={}
res['CV head (held-out, from predictions.csv)']=[p['activation_ins_buzz'].to_numpy() for p in preds]
res['shipped head on embeddings']=hl
for v in variants: res[v]=load(v)
for col in ['sensitivity','sensitivity_exclquiet']:
    print('\n##',col,'at fpr 0.005 (per fold in folds.csv order, then mean)')
    for k,a in res.items():
        r=sens(a,col); print(f'{k:45s}',' '.join(f'{x:.3f}' for x in r),'| mean',f'{np.nanmean(r):.3f}')
