import os

import config as cfg
from models.models import BaseModel

class ModelGeneralV3(BaseModel):
    modelname = "model_general_v3"
    embeddername = 'yamnet'
    digits_results = 2

    def initialize(self):
        self.embedder.initialize()

        from keras.layers import TFSMLayer
        dir_model = os.path.abspath(os.path.join(cfg.DIR_MODELS, self.modelname))
        self.model = TFSMLayer(dir_model, call_endpoint='serving_default')  # self.model defined in ABC

    def predict_embeddings(self, embeddings):
        return self.model(embeddings)['dense']

    def predict(self, audiosamples):
        """
        Generate predictions for audio data

        Args:
            audiosamples: numpy array of audio samples at self.embedder.samplerate

        Returns:
            tensor results
        """
        embeddings = self.embedder.embed(audiosamples)
        results = self.model(embeddings)['dense']

        return results