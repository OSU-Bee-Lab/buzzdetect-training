library(dplyr)

# tiny is a smoke-test subset of the already-built `medium` set: the first two
# annotations of every ident. Everything else (folds, summaries, translations)
# is derived from medium by _derive.R. Nothing here reads 01_annotate, and
# snips/embeddings are the set's own — extract with `--set tiny`.

medium_dir <- '../medium'

read.csv(file.path(medium_dir, 'annotations.csv')) %>%
  group_by(source, ident) %>%
  slice_min(n = 2, order_by = start, with_ties = FALSE) %>%
  ungroup() %>%
  write.csv('annotations.csv', row.names = FALSE)

source('../_derive.R')
