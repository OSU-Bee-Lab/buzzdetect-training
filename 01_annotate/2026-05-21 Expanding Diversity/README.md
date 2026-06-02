Author: Luke Hearon
Note: this project is similar to, but distinct from "2025-06-07 night detections". That previous annotation effort was insufficiently documented and it would take a significant amount of effort to refurbish it into a fully working state. We'll start anew

We've found that model_general_v3 has a consistent spike of false positives during the night, usually tightly clustered and near midnight. While the rate of detections is never terribly high, it would be much better if the false positives were distributed evenly across the night.

The false positives can even dwarf the diel trend in the focal crop (e.g., 'Luke - Diel Drivers/2026-05-06').

It should be trivial to automatically generate training data to combat this. We know that all detections at night are false, so automatically generate labels from frames that were positive at night time. 

The downside of the automatic approach is specificity; we'll have to call these ambient_background or otherwise, but really they could be ins_trill, mech_auto_plane, etc. 