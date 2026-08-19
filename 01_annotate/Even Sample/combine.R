library(dplyr)
library(stringr)

# Combine annotations ----
#

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
    label = str_trim(label),
    label = case_when(
      str_detect(label, 'ins_buzz_pollination') ~ 'ins_buzz_pollination',
      str_detect(label, 'ins_buzz_medium') ~ 'ins_buzz_medium',
      str_detect(label, 'ins_buzz_high') ~ 'ins_buzz_high',
      str_detect(label, 'ins_buzz_low') ~ 'ins_buzz_low',

      str_detect(label, 'ins_trill') ~ 'ins_trill',

      label == 'animal_bird' ~ 'ambient_background',
      label == 'mech_farm' ~ '',
      label == 'happy 4th :)' ~ '', # :)
      label == 'unknown_rasp' ~ '',

      T ~ label
    )
  ) %>% 
  filter(label != '')

unique(annotations_combined$label) %>% sort()

write.csv(
  annotations_combined %>% 
    filter(label != ''),
  'annotations_combined.csv',
  row.names=F
)

# Assign folds ----
#
folds <- annotations_combined %>% 
  mutate(fold = dirname(ident)) %>% 
  select(ident, fold) %>% 
  unique() %>% 
  mutate(role='rotate')

write.csv(
  folds,
  'folds.csv',
  row.names=F
)

# Summaries --- 
#

summary <- annotations_combined %>% 
  left_join(folds) %>% 
  group_by(label, fold) %>% 
  mutate(duration = round(end-start), count=n()) %>% 
  summarize(
    duration = sum(duration),
    occurences = sum(count)
  ) %>% 
  tidyr::pivot_wider(
    id_cols = fold,
    names_from = label,
    values_from = c(duration, occurences),
    values_fill = 0
  )

 write.csv(
  summary,
  'summary.csv',
  row.names=F
)
