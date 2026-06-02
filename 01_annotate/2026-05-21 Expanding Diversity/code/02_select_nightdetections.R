library(dplyr)

n_snips <- 100

night_start <- 23
night_end <- 5

results <- readRDS('data/01_joined.rds') %>% 
  mutate(hour = lubridate::hour(bin_datetime)) %>% 
  filter(!(target %in% c('honey bee hive', 'yellow jacket colony','bumble bee hive'))) %>% 
  filter(frames >= 312) # just want to make sure we have the full audio

results_selected <- results %>% 
  filter((hour >= night_start) | (hour < night_end))  %>% 
  slice_max(order_by=detections_ins_buzz, n=n_snips) %>% 
  mutate(
    start = difftime(
      bin_datetime,
      buzzr::file_start_time(ident, posix_formats = '%y%m%d_%H%M', tz='America/New_York'),
      units='secs'
    ) %>% 
      as.integer(),
    end = start + 300
  ) %>% 
  select(
    ident,
    start,
    end
  )

write.csv(
  results_selected,
  'data/02_selections_night.csv',
  row.names=F
)
