dir_annotations <- '01_annotate'
dir_sets <- '02_set'

# Per-machine paths (buzzdetect export destination, external audio drives, ...)
# that don't belong in a tracked file: they differ machine to machine and mean
# nothing to anyone else's checkout. Lives at paths.local.json (gitignored);
# paths.local.example.json documents the shape. Missing file or missing key
# both just mean "unset" -- callers decide whether that's a fallback or a
# stop() telling the operator to set it.
path_local <- function(key, default = NULL) {
  path_file <- here::here('paths.local.json')
  if (!file.exists(path_file)) {
    return(default)
  }

  node <- jsonlite::fromJSON(path_file, simplifyVector = FALSE)
  for (part in strsplit(key, '.', fixed = TRUE)[[1]]) {
    if (!is.list(node) || is.null(node[[part]])) {
      return(default)
    }
    node <- node[[part]]
  }

  node
}
