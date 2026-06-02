import os
import re
import multiprocessing

import pandas as pd
import soundfile as sf

snip_duration = 300

filetime_colname = 'snip_timestamp'
dir_data = './data'
dir_audio = '/media/server storage/experiments'
dir_out = './audio'

df = pd.read_csv(os.path.join(dir_data, '01_sniptimes.csv'))

idents = df['ident'].unique()


def snip_ident(ident):
    print(f'\n\nstarting snipping of {ident}')
    sub = df[df['ident']==ident].sort_values(filetime_colname)

    ident = re.sub( '^/', '', ident)
    path_audio = os.path.join(dir_audio, ident+'.mp3')

    track = sf.SoundFile(path_audio)

    for start_time in sub[filetime_colname]:
        print(f'snipping at {start_time}')
        track.seek(start_time*track.samplerate)
        samples = track.read(snip_duration*track.samplerate)
        path_out = os.path.join(dir_out, ident + '_s' + str(start_time) + '.mp3')
        os.makedirs(os.path.dirname(path_out), exist_ok=True)

        sf.write(
            file=path_out,
            data=samples,
            samplerate=track.samplerate
        )


with multiprocessing.Pool(processes=8) as pool:
    result = pool.map_async(func=snip_ident, iterable=idents)
    result.wait()

