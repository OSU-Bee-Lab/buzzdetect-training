library(dplyr)
library(stringr)

dir_sources <- '../../../01_annotate'

sources <- c(
  'Even Sample'
  # '2025-06-04 original annotations'
)

message(
  'sourcing from: ',
  paste(sources, collapse = ', ')
)

# sources <- list.dirs(
#   dir_sources,
#   recursive = F,
#   full.names=F
# ) %>% 
#   {.[!(.%in% c('.deprecated', '.archive', '2025-06-24 InsectSound1000'))]}


# Annotations ----
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

annotations <- lapply(sources, read_annotations) %>% 
  bind_rows() %>% 
  group_by(source, ident) %>% 
  mutate(
    duration = end-start,
    item = row_number()
  ) %>% 
  mutate(
    label = case_when(
      str_detect(label, 'ins_buzz') ~ 'ins_buzz',

      label == 'ambient_thunder' ~ 'ambient_rain',
      label == 'animal_bird_goose' ~ 'ambient_background',

      str_detect(label, 'animal_frog') ~ 'animal_frog',
      label == 'ins_cicada' ~ 'ins_trill',

      label == 'ins_buzz_rasp' ~ '', # rasping of wings on mic, ignore for now

      str_detect(label, 'mech_auto') ~ 'mech_auto',
      label == 'mech_siren' ~ 'mech_auto',

      str_detect(label, 'mech_plane') ~ 'mech_plane',

      str_detect(label, 'mech_hum') ~ 'mech_hum',
            label %in% c(
        "ambient_bang",
        'ambient_scraping',
        'ambient_rustle'
      ) ~ 'ambient_noise',

      T ~ label
    )
  ) %>% 
  group_by(ident, label) %>% 
  slice_min(n=10, order_by=start)

annotations$label %>% unique() %>% sort()

write.csv(
  annotations,
  'annotations.csv',
  row.names=F
)

# Folds ----
#

read_folds <- function(source_set){
  path_folds <- file.path(dir_sources, source_set, 'folds.csv')
  if(!file.exists(path_folds)){
    warning('folds not found for ', source_set)
    return(NULL)
  }
  
  path_folds %>% 
    read.csv(colClasses = 'character') %>% 
    mutate(.before=0, source = source_set)
}

folds <- lapply(sources, read_folds) %>% 
  bind_rows() %>% 
  select(source, ident, fold, role) 


check_ident_conflict <- function(){
  df <- folds %>% 
    group_by(ident) %>% 
    mutate(foldcount=length(unique(fold))) %>% 
    filter(foldcount>1)
  
  return(unique(df$ident))
}

conflicted_idents <- check_ident_conflict()

if(length(conflicted_idents)>0){
  annotations <- read.csv('annotations.csv')
  filter(annotations, ident %in% conflicted_idents) %>% 
    select(source, ident) %>% 
    left_join(folds, by=c('source', 'ident')) %>% 
    unique() %>% 
    arrange(ident) %>% 
    write.csv('idents_conflicted.csv', row.names=F)
  
  stop('idents found in multiple folds.\nSee annotations_conflicted.csv for more information.\n', paste(conflicted_idents, collapse='\n'))
}

write.csv(
  folds,
  file.path('folds.csv'),
  row.names=F
)


# Summaries ----
#
annotations <- read.csv('annotations.csv')

summary_per_fold <- annotations %>% 
  left_join(folds, by = c('source', 'ident')) %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold) %>% 
  summarize(
    volume = sum(duration)
  )

write.csv(
  summary_per_fold,
  'summary_per_fold.csv',
  row.names=F
)

summary_per_class <- annotations %>% 
  left_join(folds, by = c('source', 'ident')) %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(
    id_cols = label,
    names_from = fold,
    values_from=volume,
    values_fill = 0
  ) %>% 
  arrange(label)

write.csv(
  summary_per_class,
  'summary_per_class.csv',
  row.names=F
)


