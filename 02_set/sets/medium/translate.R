library(dplyr)
library(stringr)

# Translations ----
#
# Per-set, and generated from this set's own annotations, so every label the set
# actually carries gets a row and nothing else does. The mapping rules live here
# rather than in the CSVs; edit them and rerun. Hand-edits to the written files
# are overwritten.

annotations <- read.csv('annotations.csv')

dir_translations <- 'translations'
dir.create(dir_translations, showWarnings = FALSE)

translation_blank <- annotations %>% 
  select(from = label) %>% 
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
  file.path(dir_translations, 'blank.csv'),
  row.names=F
)

translation_raw <- translation_blank %>% 
  mutate(
    to = case_when(
      str_detect(from, 'buzz') ~ 'ins_buzz',
      T ~ 'nonbuzz'
    )
  )


write.csv(
  translation_raw,
  file.path(dir_translations, 'raw.csv'),
  row.names=F
)


translation_general <- translation_blank %>% 
  mutate(
    to = case_when(
      # Background sounds
      from %in% c('ambient_wind') ~ 'ambient_noise',

      str_detect(from, 'music') ~ 'ambient_music',

      # this is somewhere between wind chimes and construction backup beeps...not really sure.
      # Or maybe the high-pitch resonsance of machinery? It's close enough to windchimes I'm folding
      # it into music
      from == 'ambient_ringing' ~ 'ambient_music',

      from %in% c(
        'ambient_bang',
        'ambient_noise',
        'ambient_rustle',
        'ambient_scraping',
        'ambient_scrape',
        'ambient_squeak'
      ) ~ 'ambient_noise',
      
      str_detect(from, '^animal_') ~ 'animal', 

      # Insects
      str_detect(from, '^ins_buzz') ~ 'ins_buzz',
      from %in% c('ins_trill', 'ins_cicada') ~ 'ins_trill',

      # Mechanical
      str_detect(from, '^mech_auto') ~ 'mech_auto',
      from %in% c('mech_siren', 'mech_train', 'mech_combine') ~ 'mech_auto',

      str_detect(from, '^mech_plane') ~ 'mech_plane',
      
      # Weird hums; mostly these appear to be from traffic road noise - not for anything that's hummy
      from %in% c('mech_hum', 'mech_hum_auto', 'ambient_hum_traffic', 'mech_hum_RECLASSIFY', 'mech_hum_traffic') ~ 'mech_hum',

      # Loud droning power tools; mech_ac is an outlier that I don't love
      from %in% c('mech_chainsaw', 'mech_weedwhacker', 'mech_lawnmower', 'mech_ac') ~ 'mech_tool',

      # TODO: double check if mech_drone == quad copter
      from == 'mech_drone' ~ 'ignore',
      from == 'unknown' ~ 'ignore',
      from == 'mech_hum_construction' ~ 'ignore',

      T ~ from
    )
  )

translation_general$from %>% unique() %>% sort()
translation_general$to %>% unique() %>% sort()

write.csv(
  translation_general,
  file.path(dir_translations, 'general.csv'),
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
  file.path(dir_translations, 'binary.csv'),
  row.names=F
)
