library(ggplot2)
library(dplyr)
library(stringr)
source('code/config.R')

results <- readRDS('data/01_bin.rds')


for(i in unique(results$ident)){
  stub_out <- file.path('plots', str_replace_all(i, '/', '_'))

  results_sub <- results %>% 
      filter(ident == i) %>% 
      buzzr::bin(30, T) %>% 
      filter(!is.na(bin_datetime))
  
  if(nrow(results_sub)==0){
    next
  }
  
  p_abs <- ggplot(
    results_sub,
    aes(
      x = bin_datetime,
      y = detectionrate_ins_buzz,
      color = model
    )
  ) +
    geom_path() +
    buzzr::theme_buzzr() +
    scale_color_manual(values=colors_models) +
    ggtitle(paste0('activity curves for\n', i)) +
    scale_y_continuous(expand=expansion(c(0.004, 0.05))) +
    theme(
      axis.title.y = element_blank(),
      axis.title.x = element_blank()
    )
  
  p_rel <- p_abs +
    facet_grid(rows=vars(model), scales='free_y') +
    theme(legend.position = 'none')

  ggsave(
    paste(stub_out, 'absolute.svg'),
    p_abs,
    width = 7,
    height = 3
  )

  ggsave(
    paste(stub_out, 'relative.svg'),
    p_rel,
    width = 7
  )
}

