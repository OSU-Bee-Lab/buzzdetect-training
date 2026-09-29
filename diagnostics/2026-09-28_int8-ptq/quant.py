import sys, glob, random, numpy as np, onnx, onnxruntime as ort, soundfile as sf
from onnx import helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod
D='/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/'
S=sys.argv[1]; method=sys.argv[2]; actt=sys.argv[3]   # dir, minmax|entropy|percentile, u8|s8
m=onnx.load(D+'model.onnx')
# FusedConv(Relu) -> Conv + Relu so the stock quantizer sees ai.onnx ops
new=[]
for n in m.graph.node:
    if n.op_type=='FusedConv':
        out=n.output[0]; mid=out+'_prerelu'
        c=helper.make_node('Conv',list(n.input),[mid],name=n.name,**{a.name:helper.get_attribute_value(a) for a in n.attribute if a.name!='activation'})
        r=helper.make_node('Relu',[mid],[out],name=n.name+'_relu')
        new+= [c,r]
    else: new.append(n)
del m.graph.node[:]; m.graph.node.extend(new)
seen=set()
for o in list(m.opset_import):
    if o.domain=='com.microsoft' or o.domain in seen: m.opset_import.remove(o)
    else: seen.add(o.domain)
onnx.save(m,S+'/pre.onnx')
wd={t.name:list(t.dims) for t in m.graph.initializer}
convs=[n for n in m.graph.node if n.op_type=='Conv']
# trunk convs: those fed (transitively) by ... exclude 1->1 channel filters (resampler)
trunk=[n.name for n in convs if wd.get(n.input[1],[0,0])[:2]!=[1,1] and wd[n.input[1]][1]*0+1]
mode=sys.argv[6] if len(sys.argv)>6 else 'all'
def ga(n): return next((a.i for a in n.attribute if a.name=='group'),1)
if mode.startswith('pw'):
    trunk=[n.name for n in convs if n.name in trunk and ga(n)==1 and list(next(t.dims for t in m.graph.initializer if t.name==n.input[1]))[2:]==[1,1]]
if mode.startswith('pw_skip'):
    import re as _re
    kind,k=_re.match(r'pw_skip(last|first)(\d+)',mode).groups(); k=int(k)
    pw=[n.name for n in convs if n.name in trunk]
    out=[]
    for pre in ('trunk_b0_','trunk_b1_'):
        b=[x for x in pw if x.startswith(pre)]
        out+= b[:-k] if kind=='last' else b[k:]
    trunk=out
print(len(convs),'convs,',len(trunk),'quantized, mode',mode)
class R(CalibrationDataReader):
    def __init__(s):
        random.seed(0); fs=sorted(glob.glob('/home/luke/projects/buzzdetect-training/02_set/sets/moderate/audio/snips/**/*.flac',recursive=True))
        s.fs=random.sample(fs,int(sys.argv[4]) if len(sys.argv)>4 else 120); s.i=0
    def get_next(s):
        while s.i<len(s.fs):
            f=s.fs[s.i]; s.i+=1
            x,sr=sf.read(f,dtype='float32'); x=x if x.ndim==1 else x[:,0]
            if len(x)<16000*3: continue
            return {'waveform':x[:16000*int(sys.argv[5]) if len(sys.argv)>5 else 16000*20]}
        return None
cm={'minmax':CalibrationMethod.MinMax,'entropy':CalibrationMethod.Entropy,'percentile':CalibrationMethod.Percentile}[method]
at={'u8':QuantType.QUInt8,'s8':QuantType.QInt8}[actt]
quantize_static(S+'/pre.onnx',S+f'/q_{method}_{actt}_{mode}.onnx',R(),quant_format=QuantFormat.QDQ,per_channel=True,
    weight_type=QuantType.QInt8,activation_type=at,calibrate_method=cm,nodes_to_quantize=trunk,
    extra_options={'ActivationSymmetric':False,'WeightSymmetric':True})
print('wrote',S+f'/q_{method}_{actt}_{mode}.onnx')
