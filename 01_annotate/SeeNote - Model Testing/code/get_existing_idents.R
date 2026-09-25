library(dplyr)
library(stringr)

annotation_efforts <- list.dirs('..', recursive=F, full.names=F) %>% 
  {.[!str_starts(., '\\.')]} %>% # not the .archive
  {.[.!=basename(getwd())]} # not this project

folds <- lapply(
  annotation_efforts,
  function(a){
    path_folds <- file.path('..', a, 'folds.csv')
    if(!file.exists(path_folds)){
      message(' folds.csv not found for ', a)
      return(NULL)
    }
    read.csv(path_folds)
  }
) %>% 
  bind_rows()
