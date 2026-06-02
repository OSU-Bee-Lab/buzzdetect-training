library(dplyr)
library(stringr)
library(lubridate)

snip_length <- 5*60

get_firstlast_snips <- function(start_time, duration){
  if(duration < 300){return(c(first_snip=NA_POSIXct_, last_snip=NA_POSIXct_))}
  end_time <- start_time + duration
  
  first_snip = ceiling_date(start_time, unit='hour')
  if((first_snip + snip_length) > end_time){return(c(first_snip=NA_POSIXct_, last_snip=NA_POSIXct_))}
  
  # if there wasn't enough time for a snip on the final hour, revert to last hour
  if (minute(end_time) < (snip_length/60)) {
    end_time <- lubridate::floor_date(end_time, unit = "hour") - hours(1)
  }
  
  last_snip <- end_time %>% 
    update(minute = 0, second=0)
  
  return(c(first_snip=first_snip, last_snip=last_snip))
}

idents <- read.csv('./data/raw/idents.csv')

first_last <- idents %>% 
  mutate(
    start_time = buzzr::file_start_time(ident, posix_formats = '%y%m%d_%H%M', tz='America/New_York'),
    end_time = start_time + duration
  ) %>% 
  bind_cols(
    mapply(
      get_firstlast_snips,
      .$start_time,
      .$duration,
      SIMPLIFY = F
    ) %>% 
      bind_rows()
  ) %>% 
  filter(!is.na(first_snip))


enumerate_snips <- function(ident_in, first_snip, last_snip){
  data.frame(
    ident=ident_in,
    snip_start = seq(
      from = first_snip,
      to  = last_snip,
      by = "1 hour"
    )
  )
}

snipdf <- mapply(
  enumerate_snips,
  ident_in = first_last$ident,
  first_snip = first_last$first_snip,
  last_snip =  first_last$last_snip,
  
  SIMPLIFY=F
) %>% 
  bind_rows() %>% 
  filter(lubridate::date(snip_start) == '2025-04-16') %>% 
  mutate(
    recording_start = buzzr::file_start_time(ident, posix_formats = '%y%m%d_%H%M', tz='America/New_York'),
    snip_timestamp = difftime(snip_start, recording_start, units=c('secs')) %>% 
      as.integer()
  ) %>%
  write.csv(
    './data/01_sniptimes.csv',
    row.names=F
  )

