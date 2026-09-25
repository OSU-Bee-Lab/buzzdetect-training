library(dplyr)

overwrite <- F

threshold <- -1 # liberal threshold for yamnet_moderate

dir_in <- '/Volumes/Expansion/audio_bee_detection/buzzdetect/models/yamnet_moderate/output'
dir_out <- 'data/raw/results'

paths_results <- list.files(
  dir_in,
  recursive = T,
  full.names = T,
  pattern = "_buzzdetect.csv"
)


for(path_in in paths_results){
  ident <- path_in %>% 
    stringr::str_remove(dir_in) %>% 
    stringr::str_remove('_buzz.*') %>% 
    stringr::str_remove('^/')


  path_out <- stringr::str_replace(
    path_in,
    stringr::fixed(dir_in),
    stringr::fixed(dir_out)
  ) %>% 
    tools::file_path_sans_ext() %>% 
    {paste0(., '.rds')}
  

  message("pulling ", ident)
  if(file.exists(path_out) & (!overwrite)){
    message(ident, ' already exists; skipping')
    next
  }

  df <- buzzr::read_results(
    path_in,
    posix_formats = c('%y%m%d_%H%M'),
    tz = 'America/New_York'
  ) %>% 
    buzzr::call_detections(thresholds=c(ins_buzz=threshold)) %>% 
    buzzr::bin(5) %>% 
    mutate(.before=0, ident, time_common = buzzr::commontime(bin_datetime))
  
  dir.create(dirname(path_out), recursive=T, showWarnings=F)

  saveRDS(df, path_out)
}


