library(dplyr)

# Shared build step for sets derived from an already-built `medium`.
#
# The caller (a derived set's build.R) must, before sourcing this:
#   - set its working directory to its own set dir,
#   - set `medium_dir` to the path of the built `medium` set (e.g. '../medium'),
#   - write its own `annotations.csv` — medium's annotations, subset however the
#     set wants (fewer idents, fewer rows per ident, a source filter, ...).
#
# Given that subset, this regenerates the rest of the set's annotation-level
# files *from medium*, never from 01_annotate:
#   - folds.csv    — medium's rows for the idents the subset keeps
#   - summary_*.csv — recomputed over the subset
#   - translations/ — medium's translation CSVs, rows trimmed to labels the
#                     subset still carries (the mapping rules stay in
#                     medium/translate.R; nothing is remapped here)
#
# Snips and embeddings are untouched — a derived set points `--set` at its own
# dir for extraction and gets its own audio/ and embeddings/ as usual.

if (!exists('medium_dir')) stop('_derive.R: set `medium_dir` before sourcing')

annotations <- read.csv('annotations.csv')
present_idents <- annotations %>% distinct(source, ident)
present_labels <- sort(unique(annotations$label))

# Folds ----
folds <- read.csv(file.path(medium_dir, 'folds.csv'), colClasses = 'character') %>%
  semi_join(present_idents, by = c('source', 'ident'))

missing <- anti_join(present_idents, folds, by = c('source', 'ident'))
if (nrow(missing) > 0) {
  stop('idents in the subset with no fold in medium:\n',
       paste(missing$ident, collapse = '\n'))
}

write.csv(folds, 'folds.csv', row.names = FALSE)

# Summaries ----
annotated <- annotations %>%
  left_join(folds, by = c('source', 'ident')) %>%
  mutate(duration = round(end - start))

annotated %>%
  group_by(fold) %>%
  summarize(volume = sum(duration)) %>%
  write.csv('summary_per_fold.csv', row.names = FALSE)

annotated %>%
  group_by(role) %>%
  summarize(folds = n_distinct(fold), volume = sum(duration)) %>%
  write.csv('summary_per_role.csv', row.names = FALSE)

annotated %>%
  group_by(fold, label) %>%
  summarize(volume = sum(duration), .groups = 'drop') %>%
  tidyr::pivot_wider(id_cols = label, names_from = fold,
                     values_from = volume, values_fill = 0) %>%
  arrange(label) %>%
  write.csv('summary_per_class.csv', row.names = FALSE)

# Translations ----
dir.create('translations', showWarnings = FALSE)
for (f in list.files(file.path(medium_dir, 'translations'), pattern = '\\.csv$')) {
  read.csv(file.path(medium_dir, 'translations', f)) %>%
    filter(from %in% present_labels) %>%
    write.csv(file.path('translations', f), row.names = FALSE)
}
