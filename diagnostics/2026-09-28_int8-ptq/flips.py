import sys, csv, glob, re, random, json, numpy as np, soundfile as sf, onnxruntime as ort
from collections import defaultdict
R='/home/luke/projects/buzzdetect-training/'
CL=json.load(open('/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/config_model.json'))['classes']
ref_p=sys.argv[1]; cands=sys.argv[2:]
rot={r['ident'] if False else r['fold'] for r in csv.DictReader(open(R+'02_set/sets/moderate/folds.csv')) if r['role']=='rotate'}
ann=defaultdict(list)
for r in csv.DictReader(open(R+'models/v4-ft-ps-e60-moderate/annotations.csv')):
    if r['label'].startswith('ins_buzz') and r['label']!='ins_buzz_medium_background': ann[r['ident']].append((float(r['start']),float(r['end'])))
random.seed(1); items=[]  # (fold, audio, has_buzz)
byfold=defaultdict(lambda:([],[]))
for f in glob.glob(R+'02_set/sets/moderate/audio/snips/**/*.flac',recursive=True):
    m=re.search(r'snips/(.*)/snip_([\d.]+)_([\d.]+)\.flac$',f); ident,s,e=m.group(1),float(m.group(2)),float(m.group(3))
    fold='/'.join(ident.split('/')[:-1])
    if fold not in rot or e-s<3: continue
    has=any(a<e and b>s for a,b in ann.get(ident,[]))
    byfold[fold][0 if has else 1].append(f)
for fold,(pos,neg) in byfold.items():
    for f in random.sample(pos,min(25,len(pos))): items.append((fold,f,1))
    for f in random.sample(neg,min(15,len(neg))): items.append((fold,f,0))
items.append(('FIXTURE',R+'04_deploy/fixtures/230808_1208_s89520.flac',1))
print(len(items),'clips over',len(byfold),'folds',flush=True)
def sess(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=4; return ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
def load(f):
    x,sr=sf.read(f,dtype='float32'); x=x if x.ndim==1 else x[:,0]
    if sr!=16000:
        import subprocess; x=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',f,'-ac','1','-ar','16000','-f','f32le','-'],capture_output=True).stdout,dtype='float32')
    return x[:16000*120]
import os,pickle
cache=os.path.join(os.environ.get('INT8_WORK','.local/int8-ptq'),'flips_cache.pkl')
os.makedirs(os.path.dirname(cache),exist_ok=True)
if os.path.exists(cache): audio,ref=pickle.load(open(cache,'rb'))
else:
    audio=[(fo,load(f),h) for fo,f,h in items]
    sr_=sess(ref_p); ref=[sr_.run(None,{'waveform':a})[0] for _,a,_ in audio]
    pickle.dump((audio,ref),open(cache,'wb'))
b=CL.index('ins_buzz')
print(f'reference: {sum(int((r[:,b]>0).sum()) for r in ref)} ins_buzz detections in {sum(len(r) for r in ref)} frames')
for c in cands:
    s=sess(c); out=[s.run(None,{'waveform':a})[0] for _,a,_ in audio]
    print('\n##',c.split('/')[-1])
    tot=defaultdict(lambda:[0,0,0,0])  # ref_pos, gained, lost, frames
    perfold=defaultdict(lambda:[0,0,0])
    for (fo,_,_),r,q in zip(audio,ref,out):
        for k in range(len(CL)):
            rp,qp=r[:,k]>0,q[:,k]>0
            t=tot[CL[k]]; t[0]+=rp.sum(); t[1]+=(~rp&qp).sum(); t[2]+=(rp&~qp).sum(); t[3]+=len(rp)
        rp,qp=r[:,b]>0,q[:,b]>0; p=perfold[fo]; p[0]+=rp.sum(); p[1]+=(~rp&qp).sum(); p[2]+=(rp&~qp).sum()
    print(f'{"class":22s} ref+  gained lost   (of frames)')
    b_=tot[CL[b]] if False else None
    for k,t in sorted(tot.items(),key=lambda kv:-(kv[1][1]+kv[1][2])):
        if t[0] or t[1] or t[2]: print(f'{k:22s} {t[0]:5d} {t[1]:5d} {t[2]:5d}   {t[3]}')
    print('ins_buzz by fold: ref+ / gained / lost')
    for fo,p in perfold.items(): print(f'  {fo[-40:]:40s} {p[0]:4d} {p[1]:4d} {p[2]:4d}')
    d=np.concatenate([np.abs(q[:,b]-r[:,b]) for r,q in zip(ref,out)]); print(f'ins_buzz logit |d|: mean {d.mean():.3f} p99 {np.percentile(d,99):.3f} max {d.max():.3f}')
