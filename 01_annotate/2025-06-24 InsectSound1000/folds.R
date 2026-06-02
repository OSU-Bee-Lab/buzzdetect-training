library(dplyr)
library(stringr)

annotations <- read.csv('annotations_combined.csv')

assign_folds <- function(filenames, num_folds = 4) {
  # Create hash of each filename and convert to fold assignment
  fold_assignments <- sapply(filenames, function(filename) {
    hash_value <- digest::digest(filename, algo = "md5")
    hash_int <- sum(utf8ToInt(hash_value))
    fold <- (hash_int %% num_folds) + 1
    
    return(fold)
  })
  
  fold_assignments <- c('train', 'train', 'train', 'validate')[fold_assignments]
  
  return(fold_assignments)
}

folds <- annotations %>% 
  select(ident) %>% 
  unique() %>% 
  mutate(
    fold=assign_folds(ident)
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