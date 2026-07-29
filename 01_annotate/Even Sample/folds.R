library(dplyr)
library(stringr)

annotations <- read.csv('annotations_combined.csv')

folds <- annotations %>% 
  select(ident) %>% 
  unique() %>% 
  mutate(
    ident_recorder = dirname(ident),
    fold = str_replace_all(ident_recorder, '/', '_')
  )

write.csv(
  folds,
  'folds.csv',
  row.names=F
)

summary_per_fold <- annotations %>% 
  left_join(folds, by = 'ident') %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold) %>% 
  summarize(
    volume = sum(duration)
  )

write.csv(
  summary_per_fold,
  'summary_per_fold.csv',
  row.names=F
)

summary_per_class <- annotations %>% 
  left_join(folds, by = 'ident') %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(
    id_cols = label,
    names_from = fold,
    values_from=volume,
    values_fill=0
  ) %>% 
  arrange(label)

write.csv(
  summary_per_class,
  'summary_per_class.csv',
  row.names=F
)
 