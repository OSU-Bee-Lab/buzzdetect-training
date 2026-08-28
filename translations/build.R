library(dplyr)
library(stringr)

source(here::here('config.R'))

dir_out <- here::here('translations')

# Every raw label observed across every annotation effort.
annotations <- list.dirs(here::here(dir_annotations), recursive=F) %>%
  {.[!str_detect(basename(.), '^\\.')]} %>%   # literally tell me why list.dirs doesn't have the same all.files argument that list.files already has
  file.path('annotations_combined.csv') %>%
  {.[file.exists(.)]} %>%
  lapply(read.csv) %>%
  bind_rows()

if (nrow(annotations) == 0) {
  stop('no annotations_combined.csv found under ', here::here(dir_annotations),
       '; run 01_annotate/MAKE.R first')
}


translation_blank <- annotations %>%
  select(from=label) %>%
  unique() %>%
  arrange(from) %>%
  mutate(
    to = case_when(
      tolower(from) == 'ignore' ~ 'IGNORE',
      tolower(from) == 'exclude' ~ 'EXCLUDE',
      T ~ ''
    )
  )

write.csv(
  translation_blank,
  file.path(dir_out, 'blank.csv'),
  row.names=F
)

translation_general <- translation_blank %>%
  mutate(
    to = case_when(
      # Background sounds
      from %in% c('ambient_wind') ~ 'ambient_background',

      str_detect(from, 'music') ~ 'ambient_music',

      from %in% c(
        'ambient_bang',
        'ambient_noise',
        'ambient_rustle',
        'ambient_scraping',
        'ambient_scrape',
        'ambient_squeak'
      ) ~ 'ambient_noise',
      
      str_detect(from, '^animal_') ~ 'animal', 

      # Weather
      from %in% c('ambient_rain', 'ambient_thunder') ~ 'ambient_rain',
      from == 'human' ~ 'human',

      # Insects
      str_detect(from, '^ins_buzz') ~ 'ins_buzz',
      from %in% c('ins_trill', 'ins_cicada') ~ 'ins_trill',

      # Mechanical
      str_detect(from, '^mech_auto') ~ 'mech_auto',
      from %in% c('mech_siren', 'mech_train', 'mech_combine') ~ 'mech_auto',
      # testing shows that mech_plane usually cannibalizes mech_auto - cars will activate mech_plane instead of mech_auto;
        # I believe this is because they're close enough that the model finds the easy win of just "when in doubt, always pick one"
        # and perhaps we have more planes in the training set than cars. That's fine, let's just conflate these.
      str_detect(from, '^mech_plane') ~ 'mech_auto',
      
      # Weird hums; mostly these appear to be from traffic road noise - not for anything that's hummy
      from %in% c('mech_hum', 'mech_hum_auto', 'ambient_hum_traffic', 'mech_hum_RECLASSIFY', 'mech_hum_traffic') ~ 'mech_hum',

      # Loud droning power tools
      from %in% c('mech_chainsaw', 'mech_weedwhacker', 'mech_lawnmower') ~ 'mech_tool',

      # TODO: double check if mech_drone == quad copter
      from == 'mech_drone' ~ 'ignore',
      from == 'unknown' ~ 'ignore',

      T ~ from
    )
  )

annotations %>% 
  left_join(
    translation_general %>% 
      rename('label'='from')
  ) %>% 
  pull(to) %>% 
  unique() %>% 
  sort()


write.csv(
  translation_general,
  file.path(dir_out, 'general.csv'),
  row.names=F
)


translation_binary <- translation_blank %>%
  mutate(
    to = case_when(
      str_detect(from, 'ins_buzz') ~ 'ins_buzz',
      str_detect(tolower(from), 'exclude') ~ 'EXCLUDE',

      T ~ 'nonbuzz'
    )
  )

write.csv(
  translation_binary,
  file.path(dir_out, 'binary.csv'),
  row.names=F
)
