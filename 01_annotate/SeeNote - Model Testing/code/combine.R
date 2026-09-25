library(dplyr)
library(stringr)

dir_audio <- 'data/raw/audio'

paths_audio <- list.files(
  dir_audio,
  recursive = T,
  full.names = T
) %>% 
  {.[tools::file_ext(.) %in% c('mp3', 'wma')]}

idents <- paths_audio %>% 
  str_remove(dir_audio) %>% 
  str_remove('^/') %>% 
  tools::file_path_sans_ext()


folds <- data.frame(ident=idents) %>% 
  mutate(role = 'test')

write.csv(
  folds,
  'folds.csv',
  row.names=F
)
