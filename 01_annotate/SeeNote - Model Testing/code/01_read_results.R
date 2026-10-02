library(dplyr)
library(buzzr)
library(stringr)

dir_raw <- 'data/raw/results'
models <- list.dirs(dir_raw, F, F) %>% 
  {.[!str_detect(., '^\\.')]}


# The following thresholds give a correlation with heavy_v1 that passed through 0.0
  # I'm not sure that's a perfect method, but e.g. balanced_v1 was consistently above the 1:1 line
  # even though it was linearly correlated - id est, subtracting 0.6 from the activation
  # would have produced results much more similar to heavy_v1
thresholds <- c(
  fast_v1 = 0.5,
  balanced_v1 = 0.3, 
  heavy_v1 = 0,
  model_general_v3 = -1.2,
  yamnet_moderate_fp32 = -1.7
)  %>%
  tibble::enframe(name = "model", value = "threshold")

results_raw <- lapply(
  models,
  function(m){
    buzzr::read_directory(
      file.path(dir_raw, m),
      posix_formats = '%y%m%d_%H%M',
      tz = 'America/New_York',
      return_ident = T,
      workers = 1
    ) %>% 
      mutate(.before=0, model=m)
  }
) %>% 
  bind_rows()

results_raw$model <- factor(results_raw$model, levels = thresholds$model)

saveRDS(
  results_raw,
  'data/01_raw.rds'
)


results_bin <- results_raw %>% 
  left_join(thresholds) %>% 
  mutate(detections_ins_buzz = activation_ins_buzz > threshold) %>% 
  select(!threshold) %>% 
  bin(1, T)

saveRDS(
  results_bin,
  'data/01_bin.rds'
)
