# SeeNote Model Testing
This project applies buzzdetect models to novel audio files for a full deep-dive on model performance.

Ideally, these files are all in the "Test" role, so we aren't just looking at overfitting.
The BUILD.r script produces a folds.csv in order to reserve these idents for the test set.
The training pipeline should fail loudly if one of these files has been chosen for training. 