library(dplyr)
library(stringr)

dir_annotations <- 'annotations'

paths_annotations <- list.files(
    pattern = 'annotation.csv',
    recursive = T,
    full.names=T
)

read_annotation <- function(path_in){
    # these annotations are already in file-time not snip-time, so no
    # reinterpretation needed
    ident <- path_in %>% 
        str_remove('^./') %>% 
        str_remove(dir_annotations) %>% 
        str_remove('^/') %>% 
        str_remove('_annotation.csv')
    df <- read.csv(path_in) %>% 
        mutate(.before=0, ident=ident)

    return(df)
}

annotations_combined <- lapply(paths_annotations, read_annotation) %>% 
    bind_rows() %>% 
    arrange(ident, start) %>% 
    filter(!(
        label %in% c(
            # insufficiently classified
            'RECLASSIFY',
            'mech_traffic_RECLASSIFY',
            'mech_RECLASSIFY',
            'mech_auto_RECLASSIFY',
            
            # lol
            'mech_plane_car',

            # one-off events
            'static',
            'unknownClicking'
        )
    )) %>% 
    mutate(
        ident=str_remove_all(ident, '^/'),
        
        ident = case_when(
            str_detect(ident, fixed('Karlan Forrester - Soybean Attractiveness/L68')) ~ str_replace(ident, '/L68', '/L6-8'),
            str_detect(ident, 'Bee Audio 2022 Original/8-8-22_Marysville') ~ str_replace(ident, 'Bee Audio 2022 Original/8-8-22_Marysville', 'OSPT/2022/2022-08-02'),
            T ~ ident
        ),

        label = case_when(
            label == 'bird_goose' ~ 'animal_bird_goose',
            label == 'frog_tree' ~ 'animal_frog_tree',
            label == 'frog_pickerel' ~ 'animal_frog_pickerel',
            label %in% c('ambient_day', 'ambient_night') ~ 'ambient_background',
            T~label
        )
    ) %>% 
    arrange(ident, start)

write.csv(
    annotations_combined,
    'annotations_combined.csv',
    row.names=F
)
