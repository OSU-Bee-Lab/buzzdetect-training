import soundfile as sf
import numpy as np
import pandas as pd
import os
import re
import multiprocessing

dir_audio_raw = '../../audio' # path to external hard drive
dir_audio_out = 'audio_selected'

selections = pd.read_csv('data/02_selections.csv')
selections['ident'] = [re.sub('_buzzdetect.csv', '', p) for p in selections.path]
selections['ident'] = [re.sub('^/', '', p) for p in selections.ident]


def copy_compressed(ident):
    path_audio = os.path.join(dir_audio_raw, ident + '.wav')
    track = sf.SoundFile(path_audio)
    samples = track.read()
    samples_mono = np.mean(samples, -1)

    path_out = os.path.join(dir_audio_out, ident + '.mp3')
    os.makedirs(os.path.dirname(path_out), exist_ok=True)

    sf.write(
        file=path_out,
        data=samples_mono,
        samplerate=track.samplerate
    )


if __name__ == '__main__':
    with multiprocessing.Pool(processes=8) as pool:
        result = pool.map_async(func=copy_compressed, iterable=selections.ident)
        result.wait()
    
    print("Processing complete.")
