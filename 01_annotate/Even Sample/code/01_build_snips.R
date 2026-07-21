library(dplyr)


dir_audio <- '/Volumes/Expansion/audio_bee_detection/experiments'
idents <- read.csv('data/raw/idents.csv') %>% 
  filter(ident_recorder != '')

explode_file <- function(path_audio){
  if(!file.exists(path_audio)){stop(paste('missing', path_audio))}
  runtime <- (file.size(path_audio)/6000)
  if(is.na(runtime)){stop(paste('invalid runtime for', path_audio))}

  # tz pinned so results are reproducible across machines and don't break on
  # DST-transition dates (a local "day" with 23/25 hours would fail the
  # nrow == 24 check below). The recorder filenames are local wall-clock for
  # this site, hence America/New_York. tz propagates through seq/ceiling_date/
  # difftime/lubridate::date.
  file_start <- path_audio  %>%
    basename() %>%
    stringr::str_extract("\\d{6}_\\d{4}\\.mp3$") %>%
    as.POSIXct(format="%y%m%d_%H%M", tz="America/New_York")

  if(is.na(file_start)){stop(paste('could not parse datetime from', path_audio))}

  file_end = file_start + runtime
  first_hour = lubridate::ceiling_date(file_start, "hour")

  if(file_end < first_hour){
    return(NULL)
  }

  data.frame(
    path = path_audio,
    ident = path_audio %>% 
      stringr::str_remove(dir_audio) %>% 
      stringr::str_remove('^/') %>% 
      tools::file_path_sans_ext(),
    start_datetime = seq(first_hour, file_end, by = '1 hour')
  ) %>% 
    mutate(
      start_filetime = difftime(start_datetime, file_start, units = "secs") %>% 
        as.integer()
    )
}

snip_recorder <- function(ident_in, date_extract){
  paths_audio = file.path(dir_audio, ident_in) %>%
    list.files(full.names=T, pattern="\\.mp3$")

  snips <- lapply(
    paths_audio,
    explode_file
  ) %>% 
    bind_rows() %>% 
    filter(lubridate::date(start_datetime) == date_extract)
  
  # most of these, I verified 
  if(nrow(snips) != 24){stop(paste("Bad number of snips for ident:", ident_in))}
  return(snips)
}

snips <- mapply(
  snip_recorder,
  ident_in = idents$ident_recorder,
  date_extract = idents$date_extract,
  SIMPLIFY=F
) %>% 
  bind_rows()

write.csv(
  snips,
  'data/01_snips.csv',
  row.names=F
)
