library(dplyr)
library(ggplot2)

dir_results <- 'data/raw/results'

df <- buzzr::read_directory(dir_results, workers=1)
df$deployment <- dirname(df$ident)

# ahhhh man that's way too many to plot each...
df_deployments <- df %>% 
  group_by(deployment, time_common) %>% 
  buzzr::summarize_detections()


# just for fun
df_grand <- df_deployments %>% 
  group_by(time_common) %>% 
  summarize(detectionrate_ins_buzz = mean(detectionrate_ins_buzz))

ggplot(
  df_grand,
  aes(
    x = time_common,
    y = detectionrate_ins_buzz
  )
) +
  geom_path()


  
night_positives <- df_deployments %>% 
  filter(!between(time_common, as.POSIXct('2000-01-01 05:00'), as.POSIXct('2000-01-01 22:00'))) %>% 
  group_by(deployment) %>% 
  summarize(night_detectionrate = mean(detectionrate_ins_buzz)) %>% 
  arrange(desc(night_detectionrate))

ggplot(
  df_deployments %>% 
    filter(deployment == night_positives$deployment[1]),
  aes(
    x = time_common,
    y = detectionrate_ins_buzz
  )
) +
  geom_path()


ggplot(
  df %>% 
    filter(deployment == night_positives$deployment[1]) %>% 
    filter(!between(time_common, as.POSIXct('2000-01-01 05:00'), as.POSIXct('2000-01-01 22:00'))),
  aes(
    x = bin_datetime,
    y = detections_ins_buzz,
    color = ident
  )
) +
  geom_path()
