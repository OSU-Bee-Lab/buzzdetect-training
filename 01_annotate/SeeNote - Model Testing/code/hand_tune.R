library(dplyr)
library(ggplot2)

results_raw <- readRDS('data/01_raw.rds')

idents <- results_raw$ident %>% unique()

foo <- results_raw %>% 
  filter(model == 'yamnet_moderate_fp32', ident =="Luke - Diel Drivers/2026-05-06/1_95/260508_2106") %>% 
  buzzr::call_detections(c(ins_buzz = -1.7)) %>% 
  buzzr::bin(30, T)

ggplot(
    foo,
    aes(
      x = bin_datetime,
      y = detectionrate_ins_buzz
    )
  ) +
    geom_path() +
    buzzr::theme_buzzr() +
    scale_y_continuous(expand=expansion(c(0.004, 0.05))) +
    theme(
      axis.title.y = element_blank(),
      axis.title.x = element_blank()
    )
   