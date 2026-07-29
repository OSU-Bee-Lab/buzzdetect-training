import os


def write_model_py(dir_model, modelname, embeddername, digits_results):
    class_name = ''.join(part.capitalize() for part in modelname.split('_'))
    model_py = f'''\
import os

import config as cfg
from models.models import BaseModel

class {class_name}(BaseModel):
    modelname = "{modelname}"
    embeddername = '{embeddername}'
    digits_results = {digits_results}

    def initialize(self):
        import keras
        self.embedder.initialize()
        dir_model = os.path.abspath(os.path.join(cfg.DIR_MODELS, self.modelname))
        self.model = keras.saving.load_model(os.path.join(dir_model, 'model.keras'), compile=False)

    def predict(self, audiosamples):
        embeddings = self.embedder.embed(audiosamples)
        return self.model(embeddings)

    def predict_embeddings(self, embeddings):
        return self.model(embeddings)
'''
    # 'w', not 'x': can_write() already guards against clobbering a finished
    # model, and this is the last artifact written — failing here would throw
    # away a completed training run over a leftover file.
    with open(os.path.join(dir_model, 'model.py'), 'w') as f:
        f.write(model_py)
