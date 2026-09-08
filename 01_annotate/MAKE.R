library(dplyr)
library(withr)
source(here::here('config.R'))

# An annotation effort is any directory under 01_annotate/ holding a combine.R.
# combine.R is the contract: it reads that effort's raw annotations and writes
# annotations_combined.csv and folds.csv (ident, fold, role) next to itself.
# Fold assignment belongs in combine.R — there is no separate folds.R step.
#
# Gating on combine.R rather than on a .Rproj matters: .Rproj files are
# gitignored, so an effort that has one on your machine may have none on
# anyone else's, and the effort would be silently skipped there.

effort_dirs <- list.dirs(here::here(dir_annotations), recursive = FALSE) %>%
  {.[!startsWith(basename(.), '.')]} %>%          # .archive, .deprecated, ...
  {.[file.exists(file.path(., 'combine.R'))]}

if (length(effort_dirs) == 0) {
  stop('no annotation efforts with a combine.R found under ', here::here(dir_annotations))
}

failed <- character()

for (d in effort_dirs) {
  message(paste0('building ', basename(d)))

  # An effort with a combine.R but no raw annotation files (its data not
  # synced) fails in a way that misreports the cause: list.files() returns
  # nothing, bind_rows() on the empty list yields a 0-column frame, and the
  # first mutate/select on a missing column is what actually errors. Check
  # up front and say the real thing.
  n_raw <- length(list.files(file.path(d, 'annotations'), recursive = TRUE,
                             include.dirs = FALSE))
  if (n_raw == 0 && dir.exists(file.path(d, 'annotations'))) {
    message('  [error] no annotation files under annotations/; is this effort\'s data synced?')
    failed <- c(failed, basename(d))
    next
  }

  ok <- tryCatch({
    with_dir(d, source('combine.R'))
    TRUE
  }, error = function(e) {
    message(paste0('  [error] combine.R: ', e$message))
    FALSE
  })

  if (!ok) {
    failed <- c(failed, basename(d))
    next
  }

  # combine.R is expected to leave both behind; a missing one is a silent
  # upstream break for every set that sources this effort.
  for (f in c('annotations_combined.csv', 'folds.csv')) {
    if (!file.exists(file.path(d, f))) {
      message(paste0('  [warn] combine.R did not write ', f))
    }
  }
}

message(paste0('\n', length(effort_dirs) - length(failed), '/', length(effort_dirs),
               ' effort(s) built'))
if (length(failed) > 0) {
  message('failed: ', paste(failed, collapse = ', '))
}
