read_annotations_audacity <- function(dir_annotations){
  paths <- list.files(dir_annotations, pattern='\\.txt$', recursive=T, full.names=T)

  read_single <- function(path_in){
    ident <- path_in %>% 
      stringr::str_remove(dir_annotations) %>% 
      stringr::str_remove('^/') %>% 
      tools::file_path_sans_ext()  # assumes no suffix, per SeeNote
    
    df <- read.table(path_in, sep='\t') %>% 
      rename('start'='V1', 'end'='V2', 'label'='V3') %>% 
      mutate(.before=0, ident)
  }
  
  annotations <- paths %>% 
    lapply(read_single) %>% 
    bind_rows()
}