"""Peak GPU memory + step time for the yamnet_trunk fine-tune head at one batch size.
usage: python tools_gpu_probe.py <batch> [lr_backbone]   (run one process per batch: an OOM aborts the process)"""
import tensorflow as tf  # must come first
import sys, time, numpy as np
sys.path.insert(0, '.')
from embedders.yamnet_trunk.embedder import EmbedderYamnetTrunk
b = int(sys.argv[1]); lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
gpu = tf.config.list_physical_devices('GPU')[0]
emb = EmbedderYamnetTrunk(framehop_prop=1)
m = emb.build_head(15, lr_backbone=lr, lr_head=2e-4, dropout=0.0)
x = tf.cast(np.random.rand(b, 12288).astype('float16'), tf.float32); y = tf.cast(np.random.rand(b, 15) > .9, tf.float32)
tf.config.experimental.reset_memory_stats('GPU:0')
m.train_on_batch(x, y)  # compile/autotune
t = time.time()
for _ in range(10): m.train_on_batch(x, y)
dt = (time.time() - t) / 10
info = tf.config.experimental.get_memory_info('GPU:0')
print(f'batch {b}: {dt*1000:.0f} ms/step, {dt/b*1e6:.0f} us/frame, peak {info["peak"]/2**20:.0f} MiB', flush=True)
