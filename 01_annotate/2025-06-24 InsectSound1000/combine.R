library(dplyr)
library(stringr)

source('../utils.R')

dir_annotations <- 'annotations'

annotations <- read_annotations_audacity(dir_annotations)

write.csv(
  annotations,
  'annotations_combined.csv',
  row.names=F
)
