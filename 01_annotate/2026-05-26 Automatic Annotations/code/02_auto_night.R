library(dplyr)

threshold <- -1.2

night_start <- 23
night_end <- 5


# these might actually have buzzing at night
targets_drop <- c('honey bee hive', 'yellow jacket colony','bumble bee hive')

recorderidents_test <- read.csv('../test_idents.csv') %>% 
  filter(stringr::str_detect(ident, 'Various Opportunistic')) %>% 
  {.$ident}

dir_data <- 'data'
dir_experiment <- file.path(dir_data, 'raw/Luke - Various Opportunistic Recordings')

recorders <- file.path(dir_experiment, 'recorders.csv') %>% 
  read.csv() %>% 
  mutate(target=tolower(target))

paths_results <- list.files(
  file.path(dir_experiment, 'full'),
  full.names=T,
  recursive=T
)

pull_night <- function(path_in){
  ident <- path_in %>% 
    stringr::str_remove(file.path(dir_experiment, 'full')) %>% 
    stringr::str_remove('^/') %>% 
    paste0('Luke - Various Opportunistic Recordings/', .) %>% 
    stringr::str_remove('_buzzdetect.rds$')

  target <- filter(
    recorders,
    date_deployed == ident %>% 
    dirname() %>% 
    dirname() %>% 
    basename(),
    recorder == ident %>% 
    dirname() %>% 
    basename()
  )$target

  if(target %in% targets_drop){
    message(paste('skipping', path_in, '\nTarget is', target))
    return(NULL)
  }

  ident_recorder <- dirname(ident)

  if(ident_recorder %in% recorderidents_test){
    message(paste('skipping', path_in, '\nPart of test fold'))
  }

 buzzr::read_results(
    path_in,
    posix_formats = '%y%m%d_%H%M',
    tz='America/New_York',
    drop_filetime=F
  ) %>% 
    mutate(hour = lubridate::hour(start_datetime)) %>% 
    filter(
      (hour >= night_start) | (hour < night_end),
      activation_ins_buzz > threshold
    )  %>% 
    select(start = start_filetime, activation_ins_buzz) %>% 
    mutate(.before=0, ident)
}

results_night <- lapply(
  paths_results,
  pull_night
) %>% 
  bind_rows()


results_night$end <- results_night$start + 0.96
results_night$label <- 'ambient_night'

write.csv(
  results_night,
  'data/02_auto_night.csv',
  row.names=F
)


