library(dplyr)

idents_test <- read.csv('../test_idents.csv') %>% 
  filter(stringr::str_detect(ident, 'Various Opportunistic')) %>% 
  {.$ident}

dir_data <- 'data'
dir_experiment <- file.path(dir_data, 'raw/Luke - Various Opportunistic Recordings')

recorders <- file.path(dir_experiment, 'recorders.csv') %>% 
  read.csv()

results <- file.path(dir_experiment, 'results.rds') %>% 
  readRDS() %>% 
  filter(!(dirname(ident) %in% idents_test))  # idents_test is not a file ident, but a recorder ident

data_join <- results %>% 
  inner_join(recorders) %>% 
  mutate(target=tolower(target))

results_no_meta <- data_join %>% 
  filter(
    is.na(site) | site==''
  )

meta_no_results <- data_join %>% 
  filter(
    is.na(frames)
  )



saveRDS(
  data_join,
  file.path(dir_data, '01_joined.rds')
)
