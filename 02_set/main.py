import multiprocessing

from extract import extract_set

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

    extract_set(
        setname='lite',
        embeddername='yamnet',
        overlap_event_prop=0.2,
        framehop_prop=0.5,
        n_workers=4
    )
