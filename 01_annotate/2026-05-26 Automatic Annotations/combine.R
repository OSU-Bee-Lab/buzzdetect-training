library(dplyr)
library(stringr)

n_bee = 4 # each is ~300 frames = 1,500 frames
n_night = 1500

bee <- read.csv('data/02_auto_beehive.csv') %>% 
  filter(frames>=300) %>% 
  slice_max(order_by=detections_ins_buzz, n=n_bee) %>% 
  mutate(label = 'auto_honeybeehive', end=start + (frames*0.96)) %>% 
  select(ident, start, end, label)

night <- read.csv('data/02_auto_night.csv') %>% 
  slice_max(order_by=activation_ins_buzz, n=n_night) %>% 
  select(ident, start, label) %>% 
  mutate(end = start + 0.96, label='auto_ambient_night')

write.csv(
  bind_rows(bee, night),
  'annotations_combined.csv',
  row.names=F
)
