library(dplyr)
library(stringr)

# large is medium at a finer framehop: same sources, same folds, same
# translations, and only config_extract.json differs. So the build steps are
# medium's, sourced from here — they read and write relative to the working
# directory, so their outputs land in this set rather than medium's.

dir_steps <- '../medium'

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

source(file.path(dir_steps, 'combine.R'))     # -> annotations.csv
source(file.path(dir_steps, 'folds.R'))       # -> folds.csv
source(file.path(dir_steps, 'summarize.R'))   # -> summary_per_fold.csv, summary_per_role.csv, summary_per_class.csv
source(file.path(dir_steps, 'translate.R'))   # -> translations/
