import tensorflow as tf  # must come first
import sys, os, csv, pickle, numpy as np, pandas as pd
ROOT='/home/luke/projects/buzzdetect-training/'
sys.path[:0]=[ROOT, ROOT+'03_train']
import config as cfg
from dataset import build_fold_dataset, read_pickle_exhaustive
W=sys.argv[1]; MODEL=ROOT+'models/v4-ft-ps-e60-moderate/'
tr=pd.read_csv(MODEL+'translation.csv')
rot=[r['fold'] for r in csv.DictReader(open(ROOT+'02_set/sets/moderate/folds.csv')) if r['role']=='rotate']
head=tf.keras.models.load_model(MODEL+'model.keras',compile=False)
cls=__import__('json').load(open(MODEL+'config_model.json'))['classes']; bi=cls.index('ins_buzz')
HOP=3072
for fi,fold in enumerate(rot):
    samples=build_fold_dataset(cfg.dir_embeddings_fold('moderate','yamnet_trunk_pitchshift_depth12',fold),tr)
    pred=pd.read_csv(MODEL+'folds/'+fold+'/predictions.csv')
    cnt=pred.groupby('sample').size().to_numpy()
    assert [s.frames for s in samples]==list(cnt),(fold,'frame counts differ from predictions.csv')
    runs=[]; chunks=[]; frame_run=[]  # per-frame (run idx, pos in run)
    for s in samples:
        rel=os.path.relpath(s.path,cfg.dir_embeddings_raw('moderate','yamnet_trunk_pitchshift_depth12'))
        pa=os.path.join(cfg.dir_audio('moderate'),'sr16000_fl0.96','raw',rel)
        frames=read_pickle_exhaustive(pa); assert len(frames)==s.frames,(pa,len(frames),s.frames)
        ft=pd.read_csv(os.path.join(os.path.dirname(s.path),'frametimes.csv'))
        st=ft[ft.label==os.path.basename(s.path)[:-7]].sort_values('row')['start'].to_numpy(); assert len(st)==len(frames)
        cur=[0]
        def flush(idx):
            a=[frames[idx[0]]]+[frames[i][-HOP:] for i in idx[1:]]
            runs.append(len(idx)); chunks.append(np.concatenate(a).astype(np.float32))
        for i in range(1,len(frames)):
            if abs(st[i]-st[i-1]-0.192)<2e-3 and np.array_equal(frames[i][:-HOP],frames[i-1][HOP:]): cur.append(i)
            else: flush(cur); cur=[i]
        flush(cur)
    emb=np.concatenate([np.array(s.embeddings,dtype=np.float32) for s in samples])
    hl=np.concatenate([head(emb[i:i+256],training=False).numpy()[:,bi] for i in range(0,len(emb),256)])
    np.save(f'{W}/audio_{fi}.npy',np.concatenate(chunks)); np.save(f'{W}/runs_{fi}.npy',np.array(runs)); np.save(f'{W}/headlogit_{fi}.npy',hl)
    print(fold,'samples',len(samples),'frames',len(emb),'runs',len(runs),flush=True)
