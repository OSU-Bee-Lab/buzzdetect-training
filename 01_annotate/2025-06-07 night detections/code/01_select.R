library(dplyr)
library(buzzr)
library(parallel)
library(stringr)

dir_data <- './data/raw'
dir_results <- file.path(dir_data, 'results')

dirs_recorder <- list.files(dir_results, recursive=T, full.names=T) %>% 
  dirname() %>% 
  unique()

read_ident <- function(path){
  ident <- path %>% 
    str_remove(dir_results) %>% 
    str_remove('_buzztrim.rds')
  
  read_raw(path) %>% 
    mutate(.before=0, ident=ident)
}

data_bin <- mclapply(
  dirs_recorder,
  function(dir_recorder){
    list.files(dir_recorder, full.names=T) %>% 
      lapply(read_ident) %>% 
      bind_rows() %>% 
      bin_raw(binwidth=5)
  }
) %>% 
  bind_rows()


data_night <- data_bin  %>% 
  mutate(hour=lubridate::hour(start_bin)) %>% 
  filter(hour < 4 | hour > 20) %>% 
  mutate(
    deployment = ident %>% 
      dirname() %>% 
      dirname()
  )

data_night$start_bin_file <- difftime(
  data_night$start_bin,
  buzzr::file_start_time(data_night$ident)
) %>% 
  as.numeric()


selections_even <- data_night %>% 
  slice_max(
    order_by = detections_ins_buzz,
    n = 2,
    by = deployment,
    with_ties = F
  )

selections_targeted <- data_night %>% 
  slice_max(order_by=detections_ins_buzz, n=20, with_ties = F)


selections <- bind_rows(selections_even, selections_targeted) %>% 
  unique()

write.csv(
  selections,
  './data/01_selections.csv',
  row.names=F
)
