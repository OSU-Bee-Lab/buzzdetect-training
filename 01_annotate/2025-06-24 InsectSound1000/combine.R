library(dplyr)
library(stringr)

source('../utils.R')

dir_annotations <- 'annotations'

annotations_combined <- read_annotations_audacity(dir_annotations) %>% 
  mutate(
    label = case_when(
      str_detect(label, 'strange') ~ 'EXCLUDE',  # just safer.
      T ~ label
    )
  )

annotations_combined$label %>% unique() %>% sort()

write.csv(
  annotations_combined,
  'annotations_combined.csv',
  row.names=F
)


# Assign folds ----
#
folds <- annotations_combined %>% 
  mutate(fold = dirname(ident)) %>% 
  select(ident, fold) %>% 
  unique() %>% 
  mutate(role='train')

write.csv(
  folds,
  'folds.csv',
  row.names=F
)

# Summaries --- 
#

summary <- annotations_combined %>% 
  left_join(folds) %>% 
  group_by(label, fold) %>% 
  mutate(duration = round(end-start), count=n()) %>% 
  summarize(
    duration = sum(duration),
    occurences = sum(count)
  ) %>% 
  tidyr::pivot_wider(
    id_cols = fold,
    names_from = label,
    values_from = c(duration, occurences),
    values_fill = 0
  )

 write.csv(
  summary,
  'summary.csv',
  row.names=F
)
