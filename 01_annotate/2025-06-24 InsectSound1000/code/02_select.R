library(dplyr)
library(stringr)

dir_data <- 'data'

data <- dir_data %>% 
  file.path('01_merged.rds') %>% 
  readRDS()

selection <- data %>% 
  group_by(ident) %>% 
  summarize(ins_buzz_max = max(activation_ins_buzz)) %>% 
  filter(ins_buzz_max >= -7)

write.csv(
  selection,
  file = file.path(dir_data, '02_selections.csv'),
  row.names=F
)
