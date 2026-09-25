library(dplyr)
library(stringr)

# Combine annotations (ezpz because we're just using raw idents and full file times) ----
#
dir_annotations <- 'annotations'

paths_annotations <- list.files(
  dir_annotations,
  recursive = T,
  full.names = T
)


annotations_combined <- paths_annotations %>% 
  lapply(
    function(p){
      df <- p %>% 
        data.table::fread()

      names(df) <- c('start', 'end', 'label')

      ident <- p %>% 
        str_remove(dir_annotations) %>% 
        str_remove('^/') %>% 
        tools::file_path_sans_ext()

      df$ident <- ident

      return(df)
    }
  ) %>% 
  bind_rows() 

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
  mutate(role = 'train')

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
