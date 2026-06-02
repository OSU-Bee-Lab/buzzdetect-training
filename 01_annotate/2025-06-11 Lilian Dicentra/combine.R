library(dplyr)
library(stringr)

dir_annotations <- 'annotations'
paths_annotations <- list.files(dir_annotations, pattern='txt', full.names=T, recursive=T)

read_annotation <- function(path_in){
  time_snip_start <- path_in %>% 
    basename() %>% 
    str_extract('_s(\\d+)\\.txt', group=1) %>% 
    as.integer()

  read.table(path_in, sep='\t') %>% 
    rename(start='V1', end='V2', label='V3') %>% 
    mutate(
      .before=0,
      ident= path_in %>% 
        str_remove(dir_annotations) %>% 
        str_remove('^/') %>% 
        str_remove('_s\\d+.txt$'),

      start = start + time_snip_start,
      end = end + time_snip_start
    )
}

annotations <- lapply(paths_annotations, read_annotation) %>% 
  bind_rows() %>% 
  mutate(
    label = case_when(
      label %in% c('ambient_day', 'ambient day', 'ambient_night') ~ 'ambient_background',
      T ~ label
    )
  )

write.csv(
  annotations,
  'annotations_combined.csv',
  row.names=F
)
