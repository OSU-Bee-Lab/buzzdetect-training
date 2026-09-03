library(dplyr)

# Summaries ----
#

annotations <- read.csv('annotations.csv')
folds <- read.csv('folds.csv', colClasses = 'character')

summary_per_fold <- annotations %>% 
  left_join(folds, by = c('source', 'ident')) %>% 
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

summary_per_role <- annotations %>% 
  left_join(folds, by = c('source', 'ident')) %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(role) %>% 
  summarize(
    folds = n_distinct(fold),
    volume = sum(duration)
  )

write.csv(
  summary_per_role,
  'summary_per_role.csv',
  row.names=F
)

summary_per_class <- annotations %>% 
  left_join(folds, by = c('source', 'ident')) %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(
    id_cols = label,
    names_from = fold,
    values_from=volume,
    values_fill = 0
  ) %>% 
  arrange(label)

write.csv(
  summary_per_class,
  'summary_per_class.csv',
  row.names=F
)
