library(dplyr)

# Folds ----
#

read_folds <- function(source_set){
  path_folds <- file.path(dir_sources, source_set, 'folds.csv')
  if(!file.exists(path_folds)){
    warning('folds not found for ', source_set)
    return(NULL)
  }
  
  path_folds %>% 
    read.csv(colClasses = 'character') %>% 
    mutate(.before=0, source = source_set)
}

folds <- lapply(sources, read_folds) %>% 
  bind_rows() %>% 
  select(source, ident, fold, role)

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
