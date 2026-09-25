# Hard Negatives Annotation Effort
This annotation effort attempts to tackle the issue of "hard negatives" - non-buzz frames that show high ins_buzz activation.


## Motivation
Improvements in training data and training methods (as of 2026-09-20) have failed to correct harsh spikes of false positives from certain events. See for example, Reed - Illinois Soybean/2026-07-17/6_76/TALK/20260717_093430 starting around 2026-07-17 14:00 (mechanical noise; also other spots in this file appear to have no particular event at all yet produce high activation) and Luke - Diel Drivers/2026-05-06/1_95/260506_1924 starting around 2026-06-07 22:00 (false positives from a jet plane, even though this is well represented in the training set and the second region of plane sound just a couple minutes later does not produce detections).

It is possible this confusion is hardcoded into the YAMNet embeddings and cannot be rescued without a change in model architecture or deeper fine tuning of the YAMNet backbone. In either case, collecting additional training data will either reveal or resolve the issue.

## Method
1. Avoid test set folds. Since we want the test set to be representative of a real deployment, targeted enrichment of hard negatives isn't appropriate. It would be better if we could capture the hard negatives using the Even Sample "5 minutes at the top of the hour" approach (and Luke - Diel Drivers/2026-05-06/1_95/260506_1924 did happen to catch a hard negative), but we can't rely on blind luck. We could do a targeted annotation of the hard negatives for a test ident, then put it in its own rotating fold, but this would cause information leakage across folds.
2. Plot a variety of data looking for suspect spikes. We already know a couple folds with this issue.
3. Open the idents in SeeNote; subset for high detection rates; annotate the entire regions. 30s bin, 0.4 minimum detection rate.

N. Split hard negative folds? It *would* be nice to get a sense of new production models' performance on hard negatives without having to manually scan over results from full files. But if all of the hard negatives are in one fold, we'd be getting the performance of a model that has never been trained on the negatives. So we could additionally split hard negatives into folds based on....I'm not sure - split them arbitrarily?
4. Iterate! Use the new model, plot new audio, see what's up


## Thoughts
This may tank metrics insofar as they don't have hard negatives, but the folds that do have hard negatives may show a big gain.