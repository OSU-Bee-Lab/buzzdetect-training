# Loop Friction Log

## 2026-06-02 — find command output is too large to use directly

**What happened:** LOOP.md instructs agents to run `find 02_set/sets -name '*.pickle' | sed 's|/[^/]*$||' | sort -u` to identify available embeddings. This command produces ~300KB of output (one line per snip-level directory). The output is truncated in the context window, making it impossible to read or search as directed.

**Impact:** Agent used a modified `find -maxdepth 4 -type d` to find embedding directories instead, which showed `yamnet_combined/raw/` and `yamnet_doublerate/raw/` as existing directories — but these were empty (no pickles). This led to a failed training attempt on yamnet_combined.

**Fix suggestion:** Replace the diagnostic command with one that summarizes at the set/embedder level, e.g.:
```bash
find 02_set/sets -name '*.pickle' | sed 's|.*/embeddings/\([^/]*\)/raw/.*|\1|' | sort -u
```
Or simply: `find 02_set/sets -path '*/embeddings/*/raw' -type d | while read d; do count=$(find "$d" -name '*.pickle' | wc -l); echo "$count $d"; done | sort -rn`

This would immediately surface that yamnet_combined and yamnet_doublerate have 0 pickles.

---

## 2026-06-02 — Empty embedding directories are misleading

**What happened:** `yamnet_combined/raw/` and `yamnet_doublerate/raw/` exist as directory trees but contain no `.pickle` files. There is no flag, marker file, or README indicating they are unpopulated stubs.

**Impact:** An agent using `find -type d` to enumerate available embedders will include these, attempt training, and fail with `ValueError: no samples found`.

**Fix suggestion:** Either remove the empty directories, or add a `.empty` marker file with a note explaining why (e.g., "requires external drive to extract").

---
