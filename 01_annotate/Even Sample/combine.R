library(dplyr)
library(stringr)

file_length = 300
min_gap = 0.10
gap_label = "ambient_background"

dir_annotations <- 'data/annotations'

paths_annotations <- list.files(
  dir_annotations,
  recursive = T
)


fill_gaps <- function(df){
  coverage <- df %>% 
    arrange(start) %>% 
    mutate(
      cmax_end   = cummax(as.double(end)),
      new_group  = start > lag(cmax_end, default = -Inf),
      coverage_group = cumsum(new_group)
    ) %>% 
    group_by(coverage_group) %>% 
    summarise(start = min(start), end = max(end), .groups = "drop") %>% 
    arrange(start)
  
  gaps <- tibble(
    start = c(0, coverage$end),
    end   = c(coverage$start, file_length)
  ) %>% 
    filter(end - start >= min_gap) %>% 
    mutate(label = gap_label)
  
  bind_rows(df, gaps) %>% 
    arrange(start)
}


translate_annotation <- function(path_in){
  start_snip <- path_in %>% 
    str_extract('_s(\\d+).txt$', group=1) %>% 
    as.integer()

  ident <- path_in %>% 
    str_remove('_s\\d+.txt$')

  tryCatch(
      read.table(file.path(dir_annotations, path_in), sep='\t') %>% 
        rename('start'='V1', 'end'='V2', 'label'='V3')  %>% 
          fill_gaps() %>% 
              mutate(
                .before=0,
                ident,

                start = start + start_snip,
                end = end + start_snip
              ),

        error = function(msg){
          cat('error for ', path_in, ':\n', conditionMessage(msg), '\n\n', sep='')
          return(NULL)
        }
  )
}

annotations_combined <- paths_annotations %>% 
  lapply(translate_annotation) %>% 
  bind_rows() %>% 
  mutate(
    label = case_when(
      str_detect(label, 'ins_buzz_medium') ~ 'ins_buzz_medium',
      str_detect(label, 'ins_buzz_high') ~ 'ins_buzz_high',
      str_detect(label, 'ins_buzz_low') ~ 'ins_buzz_low',
      label == 'ins_buzz_rasp' ~ '', # rasping of wings on mic, ignore for now

      str_detect(label, 'ins_trill') ~ 'ins_trill',

      str_detect(label, 'mech_auto') ~ 'mech_auto',
      label == 'mech_siren' ~ 'mech_auto',

      str_detect(label, 'mech_plane') ~ 'mech_plane',

      str_detect(label, 'mech_hum') ~ 'mech_hum',

      
      label %in% c(
        "ambient_bang",
        'ambient_scraping',
        'ambient_rustle'
      ) ~ 'ambient_noise',
      
      label == 'animal_bird' ~ 'ambient_background',
      label == 'mech_farm' ~ '',
      label == 'ambient_thunder' ~ 'ambient_rain',
      label == 'happy 4th :)' ~ '', # :)
      label == 'unknown_rasp' ~ '',

      T ~ label
    )
  )

unique(annotations_combined$label) %>% sort()

write.csv(
  annotations_combined,
  'annotations_combined.csv',
  row.names=F
)
