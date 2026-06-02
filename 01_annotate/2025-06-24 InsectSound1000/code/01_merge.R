library(dplyr)
library(parallel)

dir_data <- 'data'

dir_results <- 'data/raw/buzzdetect_results'

results <- buzzr::read_directory(
  dir_results,
  return_ident=T,
  dir_nesting = c('species', 'date', 'section')
)

saveRDS(
  results,
  file = file.path(dir_data, '01_merged.rds')
)
