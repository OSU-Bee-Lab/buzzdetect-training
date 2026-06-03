# TODO

## Snip cache: wire in and verify

The snip cache layer was just merged (`feat/cache-snips`). Set binaries have
been cleared for a fresh run.

- [ ] Wire `extract_snips` into `02_set/main.py` (and/or root `main.py`) so
      it runs automatically before `extract_set`. Currently it must be called
      manually.
- [ ] Do a fresh extraction on the `lite` set with one embedder and verify all
      three cache levels are populated:
      1. `audio/snips/<ident>/snip_*.wav`
      2. `audio/<sr_fl>/<fold>/<ident>/<label>.pickle`
      3. `embeddings/<embeddername>/raw/<fold>/<ident>/<label>.pickle`
- [ ] Confirm embedding quality is unchanged vs. pre-snip-cache runs (spot-
      check a known ident's embedding against a saved reference, or just
      re-run a training and compare metrics).
- [ ] Once lite is confirmed, run the full `standard` set across all embedders
      and verify snips are shared (i.e. `audio/snips/` is only written once
      regardless of how many embedders run).

## Known edge case to watch

`frame_audio` raises `ValueError` if a snip ends up shorter than
`framelength_s` (e.g. annotation at the edge of a file shorter than ~1s).
The exception is unhandled — it would silently kill a worker process, dropping
that ident. Pre-existing condition (old `expand_chunk` had the same guard);
extremely unlikely with real field recordings.
