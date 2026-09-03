library(dplyr)

# Combine annotations ----
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
  group_by(source, ident) %>% 
  mutate(
    duration = end-start,
    item = row_number()
  )

annotations$label %>% unique() %>% sort()

write.csv(
  annotations,
  'annotations.csv',
  row.names=F
)
