"""Build an ensemble teacher for 05_distill from deployed member ONNX graphs.

    python tools_exp/build_ens_teacher.py <name> <member>=<onnx dir>=<model dir> ...

Each member is a probe-headed export of 04_deploy (waveform -> ... -> code -> MatMul -> Add -> Sub(centers)).
The ensemble graph runs every member's trunk on the one waveform and outputs the mean of the members' RAW
logits (centers not subtracted) over the classes all members share, minus the ensemble's own activation
centers. Those are computed as 03_train/thresholds.py computes any model's: from the rotations' per-class
predictions, here the members' averaged ones.

The head is written so teacher_onnx.find_code_tensor lands on the FIRST member's code:

    predictions = Sub(Add(Add(Add(MatMul(code_1, W_1/n), MatMul(code_2, W_2/n)), ...), mean bias), centers)

so the code-regression target of a student is member 1's code (for the deployed teacher as member 1, the
very targets its own students regressed), and only the logit targets change.

Writes <buzzdetect_dest>/<name>/{model.onnx, config_model.json} and main's models/<name>/
{config_model.json, folds.csv, members.txt}. Verifies the graph against the members on the deploy fixture.
"""
import json
import os
import subprocess
import sys

import numpy as np
import onnx
import pandas as pd
from onnx import compose, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '03_train'))
sys.path.insert(0, os.path.join(ROOT, '05_distill'))

import dpaths as D  # noqa: E402
import thresholds as TH  # noqa: E402


def head_of(model):
    """(code tensor, MatMul, Add, Sub nodes) of a 04_deploy export."""
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    sub = prod[g.output[0].name]
    add = prod[sub.input[0]]
    mm = prod[add.input[0]]
    assert (sub.op_type, add.op_type, mm.op_type) == ('Sub', 'Add', 'MatMul'), (sub.op_type, add.op_type, mm.op_type)
    return mm.input[0], mm, add, sub


def run(path_or_bytes, x):
    import onnxruntime as ort
    s = ort.InferenceSession(path_or_bytes, providers=['CPUExecutionProvider'])
    return s.run(None, {'waveform': x})


def fixture():
    r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', D.FIXTURE, '-ac', '1', '-ar', '16000',
                        '-f', 'f32le', 'pipe:1'], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def ensemble_centers(model_dirs, classes):
    """thresholds.compute on the members' averaged per-class rotation logits."""
    tables = [TH.read_fold_class_predictions(d) for d in model_dirs]
    folds = sorted(tables[0])
    assert all(sorted(t) == folds for t in tables), 'members scored different folds'
    act = [TH.ACT_PREFIX + c for c in classes]
    tgt = [TH.TARGET_PREFIX + c for c in classes]
    frames = {}
    for f in folds:
        base = tables[0][f]
        for t in tables[1:]:
            o = t[f]
            assert len(o) == len(base) and (o['sample'].values == base['sample'].values).all() \
                and (o[tgt].values == base[tgt].values).all(), f'{f}: members scored different frames'
        df = base.drop(columns=[c for c in base.columns if c.startswith((TH.ACT_PREFIX, TH.TARGET_PREFIX))])
        df = pd.concat([df, base[tgt]], axis=1)
        for a in act:
            df[a] = sum(t[f][a].values for t in tables) / len(tables)
        frames[f] = df
    return TH.compute(frames, classes, 'sample')


def main(name, members):
    names, onnx_dirs, model_dirs = zip(*[m.split('=') for m in members])
    n = len(names)
    cfgs_e = [json.load(open(os.path.join(d, 'config_model.json'))) for d in onnx_dirs]
    cfgs_m = [json.load(open(os.path.join(d, 'config_model.json'))) for d in model_dirs]
    classes = [c for c in cfgs_e[0]['classes'] if all(c in e['classes'] for e in cfgs_e)]
    for nm, e in zip(names, cfgs_e):
        print(f'{nm}: {len(e["classes"])} classes, not shared: {sorted(set(e["classes"]) - set(classes))}')
    assert 'ins_buzz' in classes

    centers, stats = ensemble_centers(model_dirs, classes)
    print('ensemble centers:', {c: centers.get(c) for c in classes})
    for c in ('ins_buzz',):
        print(f'  {c}: members {[m.get("activation_centers", {}).get(c) for m in cfgs_m]} -> ensemble {centers[c]}')
    cvec = np.array([centers.get(c, 0.0) for c in classes], np.float32)

    x = fixture()
    nodes, inits, codes, mats, bias, raws = [], [], [], [], np.zeros(len(classes), np.float32), []
    first = None
    for i, (nm, d, e) in enumerate(zip(names, onnx_dirs, cfgs_e)):
        path = os.path.join(d, 'model.onnx')
        m = onnx.load(path)
        code, mm, add, sub = head_of(m)
        init = {t.name: t for t in m.graph.initializer}
        idx = [e['classes'].index(c) for c in classes]
        W = numpy_helper.to_array(init[mm.input[1]])[:, idx]
        b = numpy_helper.to_array(init[add.input[1]])[idx]
        ctr = numpy_helper.to_array(init[sub.input[1]]).reshape(-1)[idx]
        raws.append(run(path, x)[0][:, idx] + ctr)           # the member's raw logits on the fixture
        drop = {mm.name, add.name, sub.name}
        keep = [nd for nd in m.graph.node if nd.name not in drop]
        del m.graph.node[:]
        m.graph.node.extend(keep)
        used = {t for nd in keep for t in nd.input}
        keep_i = [t for t in m.graph.initializer if t.name in used]
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep_i)
        del m.graph.output[:]
        del m.graph.value_info[:]
        pre = f'm{i + 1}/'
        m = compose.add_prefix(m, pre, rename_inputs=False, rename_outputs=False)
        for nd in m.graph.node:                                # add_prefix leaves graph inputs alone only by name
            for k, t in enumerate(nd.input):
                if t == pre + 'waveform':
                    nd.input[k] = 'waveform'
        if first is None:
            first = m
            wave_in = onnx.ValueInfoProto()
            wave_in.CopyFrom(m.graph.input[0])
            wave_in.name = 'waveform'
        nodes += list(m.graph.node)
        inits += list(m.graph.initializer)
        codes.append(pre + code)
        mats.append((W / n).astype(np.float32))
        bias += b / n

    out = None
    for i, (code, W) in enumerate(zip(codes, mats)):
        inits.append(numpy_helper.from_array(W, f'ens/W{i + 1}'))
        nodes.append(helper.make_node('MatMul', [code, f'ens/W{i + 1}'], [f'ens/mm{i + 1}'], name=f'ens/MatMul{i + 1}'))
        if out is None:
            out = f'ens/mm{i + 1}'
        else:
            nodes.append(helper.make_node('Add', [out, f'ens/mm{i + 1}'], [f'ens/sum{i + 1}'], name=f'ens/Sum{i + 1}'))
            out = f'ens/sum{i + 1}'
    inits.append(numpy_helper.from_array(bias.astype(np.float32), 'ens/bias'))
    nodes.append(helper.make_node('Add', [out, 'ens/bias'], ['ens/raw'], name='ens/BiasAdd'))
    inits.append(numpy_helper.from_array(cvec[None, :], 'ens/activation_centers'))
    nodes.append(helper.make_node('Sub', ['ens/raw', 'ens/activation_centers'], ['predictions'],
                                  name='ens/subtract_activation_centers'))
    g = helper.make_graph(nodes, name, [wave_in],
                          [helper.make_tensor_value_info('predictions', onnx.TensorProto.FLOAT, [None, len(classes)])],
                          initializer=inits)
    ops = {}
    for o in first.opset_import:
        ops[o.domain] = max(ops.get(o.domain, 0), o.version)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid(k, v) for k, v in ops.items()],
                              ir_version=first.ir_version)
    onnx.checker.check_model(model)

    dir_engine = os.path.join(D.ENGINE_MODELS, name)
    os.makedirs(dir_engine, exist_ok=True)
    path_out = os.path.join(dir_engine, 'model.onnx')
    onnx.save(model, path_out)

    got = run(path_out, x)[0]
    want = sum(raws) / n - cvec
    err = float(np.abs(got - want).max())
    print(f'fixture: {got.shape} max|ensemble - (mean of members\' raw logits - centers)| = {err:.2e}')
    assert err < 1e-4, err
    sys.path.insert(0, os.path.join(ROOT, '05_distill'))
    import teacher_onnx as T
    code = T.find_code_tensor(model)
    assert code == codes[0], (code, codes[0])
    print('code tensor (member 1):', code)

    cfg_e = dict(cfgs_e[0])
    cfg_e['classes'] = classes
    cfg_e['center_stats'] = {c: stats[c] for c in classes if c in centers}
    md = dict(cfg_e.get('metadata') or {})
    md.update(modelname_internal=name, ensemble_of=list(names))
    cfg_e['metadata'] = md
    json.dump(cfg_e, open(os.path.join(dir_engine, 'config_model.json'), 'w'), indent=2)

    dir_model = os.path.join(D.MAIN, 'models', name)
    os.makedirs(dir_model, exist_ok=True)
    folds_train = []
    for m in cfgs_m:
        folds_train += [f for f in m['folds_train'] if f not in folds_train]
    for nm, m in zip(names, cfgs_m):
        print(f'{nm}: {len(m["folds_train"])} training folds; not in member 1: '
              f'{sorted(set(m["folds_train"]) - set(cfgs_m[0]["folds_train"]))}; '
              f'member 1 only: {sorted(set(cfgs_m[0]["folds_train"]) - set(m["folds_train"]))}')
    cfg_m = {'embeddername': cfgs_m[0]['embeddername'], 'set': cfgs_m[0]['set'],
             'translation': cfgs_m[0]['translation'], 'classes': classes, 'folds_train': folds_train,
             'activation_centers': centers, 'center_stats': stats, 'ensemble_of': list(names)}
    json.dump(cfg_m, open(os.path.join(dir_model, 'config_model.json'), 'w'))
    folds = pd.concat([pd.read_csv(os.path.join(d, 'folds.csv'), dtype=str) for d in model_dirs]).drop_duplicates()
    folds.to_csv(os.path.join(dir_model, 'folds.csv'), index=False)
    with open(os.path.join(dir_model, 'members.txt'), 'w') as f:
        f.write('\n'.join(f'{nm}\t{os.path.abspath(o)}\t{os.path.abspath(d)}' for nm, o, d in zip(names, onnx_dirs, model_dirs)) + '\n')
    print(f'wrote {path_out} ({os.path.getsize(path_out) / 1e6:.1f} MB) and {dir_model}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
