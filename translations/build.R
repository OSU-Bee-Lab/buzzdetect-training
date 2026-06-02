library(dplyr)
library(stringr)

annotations <- list.dirs('annotations', recursive=F) %>% 
  {.[!str_detect(basename(.), '^\\.')]} %>%   # literally tell me why list.dirs doesn't have the same all.files argument that list.files already has
  file.path('annotations_combined.csv') %>% 
  {.[file.exists(.)]} %>% 
  lapply(read.csv) %>% 
  bind_rows()

translation_blank <- annotations %>% 
  select(label) %>% 
  unique() %>% 
  arrange(label) %>% 
  rename('from'='label') %>% 
  mutate(to='')

write.csv(
  translation_blank,
  'translations/blank.csv',
  row.names=F
)

translation_general <- translation_blank %>% 
  mutate(
    to = case_when(
      from %in% c('ambient_background', 'ambient_music') ~ 'ambient_background',
      from %in% c('ambient_bang', 'ambient_noise', 'ambient_rustle', 'ambient_scraping', 'ambient_scrape', 'ambient_squeak') ~ 'ambient_noise',
      from %in% c('ambient_rain', 'ambient_thunder') ~ 'ambient_rain',
      str_detect(from, '^animal_') ~ 'animal',  # 
      from == 'human' ~ 'human',
      str_detect(from, '^ins_buzz') ~ 'ins_buzz',
      from %in% c('ins_trill', 'ins_cicada') ~ 'ins_trill',
      str_detect(from, '^mech_auto') ~ 'mech_auto',
      from %in% c('mech_siren', 'mech_train', 'mech_combine') ~ 'mech_auto',
      str_detect(from, '^mech_plane') ~ 'mech_plane',
      from %in% c('mech_hum', 'mech_hum_auto', 'ambient_hum_traffic', 'mech_hum_RECLASSIFY', 'mech_hum_traffic', 'mech_hum_lawnmower', 'mech_hum_chainsaw') ~ 'mech_hum',
      
      # TODO: double check if mech_drone == quad copter
      from %in% c('ambient_music', 'mech_drone') ~ 'ignore',
      
      T ~ from
    )
  )

translation_general$to %>% 
  unique() %>% 
  sort()

write.csv(
  translation_general,
  'translations/general.csv',
  row.names=F
)
