library(ggplot2)
library(dplyr)
source('code/config.R')

results <- readRDS('data/01_raw.rds') %>% 
  select(!c(activation_ambient_rain, activation_human)) %>% 
  filter(
    ident %in% c(
      "Luke - Diel Drivers/2026-05-06/1_95/260508_2106",
      'Luke - Diel Drivers/2026-08-18/1_16/260818_0613',
      'Luke - Pollinator Habitat/2025-07-11/gru/18/250711_1147',
      'Reed - Illinois Soybean/2026-07-17/6_76/TALK/20260717_093430'
    )
  )

results_join <- left_join(
  filter(results, model == 'heavy_v1') %>% 
    rename('activation, heavy_v1'='activation_ins_buzz') %>% 
    select(!model),

  filter(results, model != 'heavy_v1')
)

big_ol_plot <- ggplot(
  results_join,
  aes(
    x = `activation, heavy_v1`,
    y = activation_ins_buzz,
    color = model
  )
) +
  geom_smooth() +
  buzzr::theme_buzzr() +
  coord_equal()

ggsave(
  'plots/activation_correlation.svg',
  big_ol_plot,
  width = 7,
  height = 7
)
