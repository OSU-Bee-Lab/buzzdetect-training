library(dplyr)
library(stringr)

annotations <- read.csv('annotations_combined.csv')

extract_info <- function(ident_in){
  parts <- ident_in %>% 
    str_split(pattern = '/', simplify=T)

  experiment <- parts[1]
  deployment <- paste(parts[2:(length(parts)-2)], collapse='/')
  recorder <- parts[length(parts)-1] %>% 
    as.character()

  data.frame(experiment, deployment, recorder)
}

idents <- annotations$ident %>% 
  unique()

ident_df <- idents %>% 
  lapply(extract_info) %>% 
  bind_rows() %>% 
  mutate(.before=0, ident=idents)

folds <- ident_df %>% 
  distinct(experiment, deployment) %>%  # Unique deployments per experiment
  group_by(experiment) %>%
  arrange(deployment, .by_group = TRUE) %>%
  mutate(
    n = n(),
    val_n = ceiling(n * 0.2),
    fold = if_else(row_number() <= val_n, "validate", "train")
  ) %>%
  select(-n, -val_n) %>%
  right_join(ident_df, by = c("experiment", "deployment"))


# these idents are being used for training due to high volume of night detections 
folds$fold[folds$ident=='Chia - Bee Audio 2022 Original/7-13-22_SouthCharleston/8/220713_0953'] <- 'train'
folds$fold[folds$ident=='Chia_Lin - Soybean on-farm spray audios/2023-07-25 Combs Sabine-Bigelow/23/230725_1159'] <- 'train'

write.csv(
  folds,
  'folds.csv',
  row.names=F
)


summary_per_fold <- annotations %>% 
  left_join(folds, by = 'ident') %>% 
  mutate(
    duration = round(end-start),
  ) %>% 
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
  left_join(folds, by = 'ident') %>% 
  mutate(duration = round(end-start)) %>% 
  group_by(fold, label) %>% 
  summarize(
    volume = sum(duration)
  ) %>% 
  tidyr::pivot_wider(id_cols = label, names_from = fold, values_from=volume)

write.csv(
  summary_per_class,
  'summary_per_class.csv',
  row.names=F
)