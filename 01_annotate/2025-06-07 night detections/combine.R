library(dplyr)
library(stringr)

dir_annotations <- 'annotations'

paths_annotations <- list.files(
  dir_annotations,
  full.names=T,
  recursive=T
)

read_annotation <- function(path_in){
  p <- '_s(.*).txt'
  
  start_snip <- str_extract(path_in, p, group=1) %>% 
    as.integer()
  
  ident <- path_in %>% 
    str_remove(dir_annotations) %>% 
    str_remove(p) %>% 
    str_remove('^/')
  
  read.table(path_in, sep='\t') %>% 
    rename(start='V1', end='V2', label='V3') %>% 
    mutate(
      .before=0,
      ident= ident,
      start = start+start_snip,
      end = end+start_snip
    )
}

annotations_combined <- lapply(
  paths_annotations,
  read_annotation
) %>% 
  bind_rows() %>% 
  arrange(ident, start) %>% 
  mutate(
    label = case_when(
      label == 'frog_tree' ~ 'animal_frog_tree',
      T ~ label
    )
  ) %>% 
  filter(label!='mech_unknown')

write.csv(
  annotations_combined,
  'annotations_combined.csv',
  row.names = F
)
