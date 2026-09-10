# Deployments — what each fold actually is

`folds.csv` carries only `source,ident,fold,role`. The crop, the site and the
date are nowhere in the pipeline, so a fold id says nothing about the ecology it
represents. This file is that missing column. Supplied by Luke, 2026-09-09.

Nothing reads this file. It exists because every conclusion about "hard folds"
is really a conclusion about crops, and an agent reading `1_150` has no way to
know it is looking at an apple orchard.

| fold | target | site | date |
|---|---|---|---|
| `JamesU - MustardBumbler/1_29` | mustard | — | 2024-09-04 |
| `Lily - Fit+Fast/2023_R3_Marysville/53` | soybean | Marysville | 2023-08-09 |
| `Lily Adam - One Hive/recorders/willard/2024-08-07/1_11` | pumpkin | Willard | 2024-08-08 |
| `Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143` | watermelon | Wooster | 2024-07-27 |
| `Luke - Various Opportunistic Recordings/2025-07-03/1_37` | chicory | Rothenbuhler | 2025-07-04 |
| `Luke - Diel Drivers/2026-05-06/1_95` | blueberry | Big D's Blueberry Patch | 2026-05-07 |
| `Luke - Various Opportunistic Recordings/2025-06-23/1_23` | milkweed | Rothenbuhler | 2025-06-24 |
| `Luke - Various Opportunistic Recordings/2025-08-05/31` | partridge pea | Rothenbuhler | 2025-08-06 |
| `Luke - Various Opportunistic Recordings/2025-08-12/1_114` | senna | Lynd House | 2025-08-13 |
| `Luke - Various Opportunistic Recordings/2025-08-27/48` | yellow jacket colony | Rothenbuhler | 2025-08-28 |
| `Luke - Diel Drivers/2026-04-08/1_150` | apple | Lynd Apple | 2026-04-09 |
| `Luke - Various Opportunistic Recordings/2026-07-27/1_99` | sunflower | Circle S Farms | 2026-07-28 |

## What it changes about reading results

**The 5 rotating folds (2026-09-09)** are mustard `1_29`, soybean `53`, pumpkin
`1_11` (often written "willard"), apple `1_150`, blueberry `1_95`.

The split between easy and hard folds is **ecological, not just statistical**.
The two folds that sit near 0.43 whatever we do are the mass-flowering row crops
(mustard, soybean). The three hard folds are pumpkin, apple and blueberry —
orchard, bush and vine crops with sparser and different bee assemblages. So
"`1_150` is a thin fold" understates it: it is a *different kind of deployment*,
not the same deployment with less annotation. More annotation will not convert an
apple orchard into a mustard field.

## What it changes about the CV protocol

The folds span **4 sites and 4 years (2023–2026)**, with a different crop and a
different bee community in each. That is much more heterogeneous than the sensor
networks the cross-validation literature is usually written about — Lostanlen et
al. (2019) rotate roles across 6 sensors of *identical hardware*, recorded on one
night over one ~1000 km² area. That is one deployment sampled six ways; this is
twelve deployments.

The consequence is in `notes/epoch-selection.md` on `exp/xfold-epoch`: these
numbers are a **ranking device between configs**, not a forecast of performance
on a new deployment, and no resampling scheme fixes that — it is a property of
having twelve deployments, not of the protocol. Adding more recorders *within* an
existing deployment does not help either; they are not independent folds, and
`03_train/CLAUDE.md`'s "validation is always a whole fold" invariant is the same
point.
