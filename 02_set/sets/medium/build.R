library(dplyr)
library(stringr)

dir_sources <- '../../../01_annotate'

sources <- list.dirs(
  dir_sources,
  recursive = F,
  full.names=F
) %>% 
  {.[.!='.deprecated']}


# Annotations ----
#

read_annotations <- function(source_set){
  path_annotations <- file.path(dir_sources, source_set, 'annotations_combined.csv')
  if(!file.exists(path_annotations)){
    warning('annotations_combined not found for ', source_set)
    return(NULL)
  }
  
  path_annotations %>% 
    read.csv() %>% 
    mutate(.before=0, source = source_set)
}

annotations <- lapply(sources, read_annotations) %>% 
  bind_rows() %>% 
  group_by(source) %>% 
  mutate(
    duration = end-start,
    item = row_number()
  ) %>% 
  filter(
    (duration < 100) | (stringr::str_detect(label, 'buzz')),
    (item <= 100) | (stringr::str_detect(label, 'buzz'))
  ) 

annotations$label %>% unique() %>% sort()

write.csv(
  annotations,
  'annotations.csv',
  row.names=F
)

# Folds ----
#

read_folds <- function(source_set){
  path_folds <- file.path(dir_sources, source_set, 'folds.csv')
  if(!file.exists(path_folds)){
    warning('folds not found for ', source_set)
    return(NULL)
  }
  
  path_folds %>% 
    read.csv() %>% 
    mutate(.before=0, source = source_set)
}

folds <- lapply(sources, read_folds) %>% 
  bind_rows() %>% 
  select(source, ident, fold)

check_ident_conflict <- function(){
  df <- folds %>% 
    group_by(ident) %>% 
    mutate(foldcount=length(unique(fold))) %>% 
    filter(foldcount>1)
  
  return(unique(df$ident))
}

conflicted_idents <- check_ident_conflict()

if(length(conflicted_idents)>0){
  annotations <- read.csv('annotations.csv')
  filter(annotations, ident %in% conflicted_idents) %>% 
    select(source, ident) %>% 
    left_join(folds, by=c('source', 'ident')) %>% 
    unique() %>% 
    arrange(ident) %>% 
    write.csv('idents_conflicted.csv', row.names=F)
  
  stop('idents found in multiple folds.\nSee annotations_conflicted.csv for more information.\n', paste(conflicted_idents, collapse='\n'))
}

write.csv(
  folds,
  file.path('folds.csv'),
  row.names=F
)


# Summaries ----
#
annotations <- read.csv('annotations.csv')

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

summary_per_class <- annotations %>% 
  left_join(folds, by = c('source', 'ident')) %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(id_cols = label, names_from = fold, values_from=volume)

write.csv(
  summary_per_class,
  'summary_per_class.csv',
  row.names=F
)


