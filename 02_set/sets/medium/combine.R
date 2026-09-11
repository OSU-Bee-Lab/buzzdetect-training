library(dplyr)

# Combine annotations ----
#

read_annotations <- function(source_set){
  path_annotations <- file.path(dir_sources, source_set, 'annotations_combined.csv')
  if(!file.exists(path_annotations)){
    warning('annotations_combined not found for ', source_set)
    return(NULL)
  }
  
  path_annotations %>% 
    read.csv() %>% 
    mutate(.before=0, source = source_set)
}

warning('dropping preliminary annotations from Luke - Various Opportunistic Recordings/2026-07-27/1_99')

annotations <- lapply(sources, read_annotations) %>% 
  bind_rows() %>% 
  filter(!stringr::str_detect(ident, 'Luke - Various Opportunistic Recordings/2026-07-27/1_99'))


annotations <- annotations %>% 
  group_by(source, ident) %>% 
  mutate(
    duration = end-start,
    item = row_number(),

    label = case_when(
      str_detect(label, 'animal') ~ 'animal',
      str_detect(label, 'ambient_music') ~ 'ambient_music',
      str_detect(label, 'ins_trill') ~ 'ins_trill',
      str_detect(label, 'human') ~ 'human',
      str_detect(label, 'mech_auto') ~ 'mech_auto',
      str_detect(label, 'mech_plane') ~ 'mech_plane',
      str_detect(label, 'mech_hum') ~ 'mech_hum',

      label %in% c('mech_siren', 'mech_train') ~ 'mech_auto',

      label %in% c('ambient_bang', 'ambient_scraping', 'ambient_rustle', 'ambient_wind', 'static') ~ 'ambient_noise',
      
      label %in% c(
        'mech_ac', 'mech_background', 'mech_beep', 'mech_chainsaw', 'mech_lawnmower', 'mech_drone',
        'mech_combine', 'mech_unknown', 'mech_weedwhacker'
      ) ~ 'mech_machinery',

      label %in% c('unknown', 'unknown_rasp') ~ '',

      TRUE ~ label
    )
  ) %>%
  # Drop the rows the recode above blanked. This filter used to live in
  # '01_annotate/Even Sample/combine.R' and came out with the recoding when it
  # moved here (f0ac4eb); without it a blank label reaches annotations.csv as
  # NaN, and extract.py's events_in_frame sorts a frame's labels, so NaN + str
  # raises "'<' not supported between instances of 'float' and 'str'". That
  # killed extraction for whichever idents carried one — on 2026-09-11, 8 rows
  # across 1_143, 1_95 and 1_114, i.e. 3 of the 8 rotating folds.
  filter(label != '')

annotations$label %>% unique() %>% sort()

write.csv(
  annotations,
  'annotations.csv',
  row.names=F
)
