from embedders.yamnet.params import Params
from embedders.yamnet.yamnet import yamnet_frames_model
import keras
import os
import numpy as np

dir_embedder = 'embedding/yamnet_combined'

model = yamnet_frames_model(Params())
model.layers[1].params.patch_hop_seconds = 0.96

model.load_weights(os.path.join(dir_embedder, 'weights_keras2/yamnet.h5'))

embeddings_out = model.get_layer("global_average_pooling2d").output
classifications_out = model.get_layer("activation").output

combined_out = keras.layers.Concatenate(axis=-1)([embeddings_out, classifications_out])

model_combined = keras.Model(
    inputs=model.input,
    outputs=combined_out,
    name="yamnet_combined"
)

z = np.zeros(15360*4)
pred_flat = model_combined(z)

print(pred_flat.shape)

path_out = os.path.join(dir_embedder, 'yamnet_combined.keras')
model_combined.save(path_out)