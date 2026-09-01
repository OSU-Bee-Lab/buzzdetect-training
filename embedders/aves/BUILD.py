"""Fetch the AVES-base-bio checkpoint that embedder.py loads.

The weights (~360 MB) are gitignored (embedders/.gitignore: aves/*.pt), so a
fresh clone needs this once. model_config.json is tracked and already matches
the ported config below.

Source: https://github.com/earthspecies/aves ("bio" = core + AudioSet/VGGSound
animal audio). Run from the project root:

    conda run -n buzzdetect-train python embedders/aves/BUILD.py
"""
import os
import urllib.request

BASE = "https://storage.googleapis.com/esp-public-files/ported_aves"
DEST = "embedders/aves"

files = {
    "aves-base-bio.pt": f"{BASE}/aves-base-bio.torchaudio.pt",
    "model_config.json": f"{BASE}/aves-base-bio.torchaudio.model_config.json",
}

for name, url in files.items():
    out = os.path.join(DEST, name)
    if os.path.exists(out):
        print(f"skip {out} (exists)")
        continue
    print(f"{url}\n  -> {out}")
    urllib.request.urlretrieve(url, out)

print("done")
