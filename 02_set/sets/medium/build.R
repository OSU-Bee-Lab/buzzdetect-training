library(dplyr)
library(stringr)

# Building the set is four steps, each one a file next to this one, each one
# reading its inputs off disk and writing its outputs there. Run this to build
# everything, or source a single step on its own to redo just that part.

dir_sources <- '../../../01_annotate'

sources <- c(
  'Even Sample',
  '2025-06-04 original annotations',
  '2025-06-24 InsectSound1000'
)

message(
  'sourcing from: ',
  paste(sources, collapse = ', ')
)

# sources <- list.dirs(
#   dir_sources,
#   recursive = F,
#   full.names=F
# ) %>% 
#   {.[!(.%in% c('.deprecated', '.archive', '2025-06-24 InsectSound1000'))]}

source('combine.R')     # -> annotations.csv
source('folds.R')       # -> folds.csv
source('summarize.R')   # -> summary_per_fold.csv, summary_per_class.csv
source('translate.R')   # -> translations/
