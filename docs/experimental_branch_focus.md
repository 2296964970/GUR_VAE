# Experimental Branch Focus

## Objective
- Strengthen the locate-plus-repair pipeline so that detection decisions and reconstructions jointly minimize recovery error on FDIA-contaminated measurements.

## Pain Points
- Decoder variance is effectively static, limiting how well tail NLL scores reflect true uncertainty; this undermines both detection confidence and reconstruction robustness.
- Tail scoring relies on single-step per-feature NLL thresholds without temporal aggregation or false-discovery control, making localization sensitive to noise and variance drift.
- Repair currently applies hard masks and a single-pass imputation, so misclassified features either pollute the context or remain corrupted, degrading reconstruction quality.
- Self-supervised masking during training samples independent feature drops and fails to mimic correlated or block-structured attacks, reducing robustness to realistic FDIA patterns.
- Evaluation emphasizes tail RMSE but omits precision/recall style metrics or confidence analysis, leaving localization quality under-quantified.
