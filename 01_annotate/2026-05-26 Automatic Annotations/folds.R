library(dplyr)
library(stringr)

annotations <- read.csv('annotations_combined.csv')

folds <- annotations %>% 
  select(ident) %>% 
  unique() %>% 
  mutate(
    fold='train'
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
  'summary_per_fold_buzz.csv',
  row.names=F
)

summary_per_class <- annotations %>% 
  left_join(folds, by = 'ident') %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(id_cols = label, names_from = fold, values_from=volume)

write.csv(
  summary_per_class,
  'summary_per_class_buzz.csv',
  row.names=F
)
