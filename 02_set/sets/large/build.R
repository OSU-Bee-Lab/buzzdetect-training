# large is medium at a finer framehop. The annotations, folds, summaries and
# translations are byte-identical to medium's by construction, so this build
# just copies them from the already-built `medium` set. Only
# config_extract.json differs (framehop_prop 0.2 vs 1) — that file is not
# touched here.
#
# Snips and embeddings are not copied: extract with `--set large` and this dir
# gets its own audio/ and embeddings/ at the finer hop.

medium_dir <- '../medium'

files <- c('annotations.csv', 'folds.csv',
           'summary_per_fold.csv', 'summary_per_role.csv', 'summary_per_class.csv')

for (f in files) {
  ok <- file.copy(file.path(medium_dir, f), f, overwrite = TRUE)
  if (!ok) stop('failed to copy ', f, ' from ', medium_dir, ' — is medium built?')
}

unlink('translations', recursive = TRUE)
dir.create('translations')
file.copy(list.files(file.path(medium_dir, 'translations'), full.names = TRUE),
          'translations', overwrite = TRUE)

message('large: copied annotations/folds/summaries/translations from ', medium_dir)
