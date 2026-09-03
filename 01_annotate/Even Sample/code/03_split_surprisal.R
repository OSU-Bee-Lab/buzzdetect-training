# Split a model's surprisal dir into per-snip files.
#
# 03_train writes one surprisal CSV per source-audio ident, with `start` in
# seconds into the whole original file. The annotations in data/annotations/ are
# cut into 300 s snips (see 01_build_snips.R / 02_snip.py), named
# <ident.name>_s<start_filetime>. This rebases surprisal onto that same layout so
# a snip's scores sit next to its annotation file.
#
# Output: <dir_out>/<ident.parent>/<ident.name>_s<start_filetime>.csv, `start`
# rebased to seconds into the snip. Rows outside every snip's window are dropped.

library(dplyr)

model <- 'yamnet_medium_general'
dir_surprisal <- file.path("../../models", model,"surprisal")
dir_out <-  "data/surprisal"
snip_length <- 300

if (!dir.exists(dir_surprisal)) stop(paste("missing surprisal dir:", dir_surprisal))

snips <- read.csv("data/01_snips.csv") %>%
  transmute(ident, start_filetime = as.integer(start_filetime))

idents_surprisal <- list.files(
  dir_surprisal,
  pattern = "_surprisal\\.csv$",
  recursive = TRUE
) %>%
  sub("_surprisal\\.csv$", "", .)
if (length(idents_surprisal) == 0) stop(paste("no *_surprisal.csv under", dir_surprisal))

for (ident in idents_surprisal) {
  path <- file.path(dir_surprisal, paste0(ident, "_surprisal.csv"))

  df <- data.table::fread(path)
  ident_snips <- snips %>% filter(ident == !!ident)
  if (nrow(ident_snips) == 0) {
    warning(paste("no snips in 01_snips.csv for ident:", ident))
    next
  }

  for (t0 in ident_snips$start_filetime) {
    out <- df %>%
      filter(start >= t0, start < t0 + snip_length) %>%
      mutate(start = start - t0) %>%
      arrange(start)  %>% 
      select(!label) %>% 
      rename(activation_loss = loss)

    if (nrow(out) == 0) next

    path_out <- file.path(
      dir_out,
      dirname(ident),
      sprintf("%s_s%d_buzzdetect.csv", basename(ident), t0)
    )
    dir.create(dirname(path_out), recursive = TRUE, showWarnings = FALSE)
    write.csv(out, path_out, row.names = FALSE)
  }
}
