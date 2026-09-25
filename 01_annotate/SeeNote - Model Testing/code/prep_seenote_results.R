library(dplyr)
library(buzzr)

dir_data <- 'data'
dir_results <- 'data/raw/results'

# don't have everything for model_general_v3
models <- list.dirs(dir_results, full.names = F, recursive = F)
dir_audio <- 'data/raw/audio'
dir_out <- 'data/01_results_compare'

files_audio <- list.files(dir_audio, recursive=T, full.names=T) %>% 
  {.[tools::file_ext(.)!='txt']}

idents <- stringr::str_remove(files_audio, dir_audio) %>% 
  tools::file_path_sans_ext() %>% 
  stringr::str_remove('^/')

for(i in idents){
  results <- lapply(
    models,
    function(m){
      path_results <- file.path(dir_results, m, paste0(i, '_buzzdetect.csv'))
      if(!file.exists(path_results)){return(NULL)}
      data.table::fread(file=path_results) %>% 
        select(start, activation_ins_buzz) %>% 
        mutate(.before=0, model=m)
    }
  ) %>% 
    bind_rows() %>% 
    tidyr::pivot_wider(
      id_cols = 'start',
      names_from = model,
      values_from = 'activation_ins_buzz'
    )
  
  path_out <- file.path(dir_out, paste0(i, '_buzzdetect.csv'))
  dir.create(dirname(path_out), recursive=T, showWarnings=F)
  data.table::fwrite(results, path_out)
}
