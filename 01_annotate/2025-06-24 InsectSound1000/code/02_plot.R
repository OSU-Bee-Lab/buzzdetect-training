library(dplyr)
library(stringr)
library(ggplot2)

data <- 'data/01_merged.rds' %>% 
  readRDS()

ggplot(
  data,# %>% 
    # filter(activation_ins_buzz > -7),
  aes( 
    x = activation_ins_buzz,
    color = species
  )
) +
  geom_density() +
  geom_vline(xintercept=-1.2) +
  buzzr::theme_buzzr(16) +
  facet_grid(rows=vars(species)) +
  theme(strip.text.y=element_text(angle=0, size=8), legend.position='none')


data_perfile <- data %>% 
  group_by(ident, species) %>% 
  summarize(activation_ins_buzz_mean = mean(activation_ins_buzz), activation_ins_buzz_max = max(activation_ins_buzz))

ggplot(
  data_perfile,
  aes( 
    x = activation_ins_buzz_max,
    color = species
  )
) +
  geom_density() +
  theme_minimal() +
  facet_grid(rows=vars(species), scales='free_y') +
  theme(strip.text.y=element_text(angle=0, size=8), legend.position='none')
