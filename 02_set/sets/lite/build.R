library(dplyr)

# lite is a troubleshooting subset of the already-built `medium` set: the
# "Even Sample" source only, capped at 10 annotations per ident per label.
# Folds, summaries and translations are derived from medium by _derive.R.
# Nothing here reads 01_annotate; snips/embeddings are the set's own
# (extract with `--set lite`).

medium_dir <- '../medium'

read.csv(file.path(medium_dir, 'annotations.csv')) %>%
  filter(source == 'Even Sample') %>%
  group_by(source, ident, label) %>%
  slice_min(n = 10, order_by = start, with_ties = FALSE) %>%
  ungroup() %>%
  write.csv('annotations.csv', row.names = FALSE)

source('../_derive.R')
