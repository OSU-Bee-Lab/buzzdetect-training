library(dplyr)
library(stringr)

annotation_efforts <- list.dirs('.', recursive=F, full.names=F) %>% 
  {.[!str_starts(., '\\.')]} # not the .archive

folds <- lapply(
  annotation_efforts,
  function(a){
    path_folds <- file.path(a, 'folds.csv')
    if(!file.exists(path_folds)){
      message(' folds.csv not found for ', a)
      return(NULL)
    }
    read.csv(path_folds) %>% 
      mutate(.before=0, annotation_effort=a)
  }
) %>% 
  bind_rows()



# Folds ----
#
conflicted_idents <- folds %>% 
  group_by(ident) %>% 
  filter(length(unique(role)) > 1)
