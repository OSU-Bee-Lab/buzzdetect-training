library(dplyr)
library(stringr)

n_bee = 4 # each is ~300 frames = 1,500 frames
n_night = 1500

bee <- read.csv('data/02_auto_beehive.csv') %>% 
  filter(frames>=300) %>% 
  slice_max(order_by=detections_ins_buzz, n=n_bee) %>% 
  mutate(label = 'ins_buzz', end=start + (frames*0.96)) %>% 
  select(ident, start, end, label)

night <- read.csv('data/02_auto_night.csv') %>% 
  slice_max(order_by=activation_ins_buzz, n=n_night) %>% 
  select(ident, start, label) %>% 
  mutate(end = start + 0.96, label='ambient_background')


annotations <- bind_rows(bee, night)

write.csv(
  annotations,
  'annotations_combined.csv',
  row.names=F
)



summary <- annotations %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(label) %>% 
  summarize(
    volume = sum(duration)
  )

write.csv(
  summary,
  'summary.csv',
  row.names=F
)

# Assign folds ----
#
# This effort is machine-labeled ballast, not a deployment: one fold, role
# 'train', so it always trains and is never scored. (Was folds.R until fold
# assignment moved into combine.R.)
folds <- annotations %>% 
  select(ident) %>% 
  unique() %>% 
  mutate(
    fold = 'auto',
    role = 'train'
  )

write.csv(
  folds,
  'folds.csv',
  row.names=F
)

summary_per_fold <- annotations %>% 
  left_join(folds, by = 'ident') %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold) %>% 
  summarize(
    volume = sum(duration)
  )

write.csv(
  summary_per_fold,
  'summary_per_fold_buzz.csv',
  row.names=F
)

summary_per_class <- annotations %>% 
  left_join(folds, by = 'ident') %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(id_cols = label, names_from = fold, values_from=volume)

write.csv(
  summary_per_class,
  'summary_per_class_buzz.csv',
  row.names=F
)
