library(dplyr)
library(withr)
source(here::here('config.R'))

annotation_dirs <- list.dirs(dir_annotations, recursive = F)

for (d in annotation_dirs) {
  message(paste0('building ', basename(d)))
  
  has_rproj <- length(list.files(d, pattern = '\\.Rproj$', full.names = F)) > 0
  if (!has_rproj) {
    message(paste0('  [skip] no .Rproj found in ', d, ', skipping'))
    next
  }
  
  path_combine <- file.path(d, 'combine.R')
  if (file.exists(path_combine)) {
    tryCatch(
      with_dir(d, source(path_combine)),
      error = function(e) message(paste0('  [error] combine.R in ', d, ': ', e$message))
    )
  } else {
    message(paste0('  [skip] combine.R not found in ', d))
  }
  
  path_folds <- file.path(d, 'folds.R')
  if (file.exists(path_folds)) {
    tryCatch(
      with_dir(d, source(path_folds)),
      error = function(e) message(paste0('  [error] folds.R in ', d, ': ', e$message))
    )
  } else {
    message(paste0('  [skip] folds.R not found in ', d))
  }
}