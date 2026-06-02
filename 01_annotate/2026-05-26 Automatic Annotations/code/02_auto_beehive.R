library(dplyr)

# conservative
day_start <- 8
day_end <- 20

results <- readRDS('data/01_joined.rds') %>% 
  filter(target == 'honey bee hive') %>% 
  mutate(hour = lubridate::hour(bin_datetime)) %>% 
  filter((hour >= day_start) & (hour < day_end))


results_selected <- results %>% 
  ungroup() %>% 
  mutate(
    start = difftime(
      bin_datetime,
      buzzr::file_start_time(ident, posix_formats = '%y%m%d_%H%M', tz='America/New_York'),
      units='secs'
    ) %>% 
      as.integer(),
    end = start + (frames*0.96)
  )

write.csv(
  results_selected,
  'data/02_auto_beehive.csv',
  row.names=F
)
