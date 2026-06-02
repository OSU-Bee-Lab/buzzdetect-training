library(dplyr)

n_per_target <- 2

results <- readRDS('data/01_joined.rds') %>% 
  mutate(date = lubridate::date(bin_datetime)) %>% 
  # take interesting targets
  filter(
    target %in% c(
      'partridge pea', 'senna',  # likely to have bumble bees and buzz pollination
      'yellow jacket colony', 'milkweed',
      'royal catchfly'
    )
  ) %>%   
  # make sure they have data for 5 minute bins at the top of each hour
  mutate(minute = lubridate::minute(bin_datetime)) %>% 
  filter(minute==0) %>% 
  group_by(site, date, recorder, target) %>% 
  filter(n()==24)


results_daily <- results %>% 
  # count detections
  summarize(detections = sum(detections_ins_buzz)) %>% 

  # keep only one recorder per day, prioritize highest detections (this way we don't pull audio from two recorders on the same day)
  group_by(site, date, target) %>% 
  slice_max(order_by=detections, with_ties=F, n=1)


# if there were two sites, just take the best day from each
selections_twosites <- results_daily %>% 
  group_by(target) %>% 
  filter(length(unique(site))==2) %>% 
  group_by(site, target) %>% 
  slice_max(order_by=detections, with_ties=F)

# if there was one site, take the two best days
selections_onesite <- results_daily %>% 
  group_by(target) %>% 
  filter(length(unique(site))==1) %>% 
  group_by(site, target) %>% 
  slice_max(order_by=detections, with_ties=F, n=2)


selections <- bind_rows(selections_twosites, selections_onesite) %>% 
  mutate(id=interaction(recorder, date))

results$id <- with(results, interaction(recorder, date))


results_selected <- results %>% 
  ungroup() %>% 
  filter(id %in% selections$id) %>%
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
    target,
    site,
    date,
    bin_datetime,
    ident,
    start,
    end
  )

write.csv(
  results_selected,
  'data/02_selections_even.csv',
  row.names=F
)
