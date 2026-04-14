# CAPI Scratch Classifier — Deployment Bundle

This directory holds the deployable artifacts for the over-review scratch
post-filter. Bundle layout:

```
deployment/
├── scratch_classifier_v1.pkl    # LoRA + LogReg + metadata (produced by train_final_model.py)
├── dinov2_vitb14.pth            # DINOv2 base weights (330 MB)
└── README.md                    # this file
```

## Producing a new bundle

Run on a machine with the training data + GPU:

```bash
python -m scripts.over_review_poc.train_final_model \
    --manifest datasets/over_review/manifest.csv \
    --transform clahe --clahe-clip 4.0 \
    --rank 16 --n-lora-blocks 2 --epochs 15 --alpha 16 \
    --calib-frac 0.2 \
    --output deployment/scratch_classifier_v1.pkl
```

The script prints the conformal threshold and estimated effective threshold at
the default safety multiplier. Commit these numbers + the SHA256 of the bundle
in the version log below.

## Exporting DINOv2 base weights (one-time, on machine with internet)

```bash
python -m scripts.over_review_poc.prepare_offline_model \
    --export-state-dict deployment/dinov2_vitb14.pth
```

## Deploying to production

1. `rsync -av deployment/ production-host:/opt/capi/deployment/` (or SCP whole dir).
2. On production host, ensure `server_config.yaml` points at these paths:
   ```yaml
   scratch_classifier_enabled: true
   scratch_safety_multiplier: 1.1
   scratch_bundle_path: /opt/capi/deployment/scratch_classifier_v1.pkl
   scratch_dinov2_weights_path: /opt/capi/deployment/dinov2_vitb14.pth
   ```
3. Restart `capi_server.py`.
4. Verify first NG request in logs: "ScratchClassifier loaded: rank=16 blocks=2 threshold=X.XXXX".

## Runtime override

Toggle via DB `config_params` (no restart needed):

```sql
UPDATE config_params SET param_value = 'false'
 WHERE param_name = 'scratch_classifier_enabled';
-- or
UPDATE config_params SET param_value = '1.2'
 WHERE param_name = 'scratch_safety_multiplier';
```

## Version log

| Version | Built at | git_commit | Conformal thr | Notes |
|---------|----------|------------|---------------|-------|
| v1      | TBD      | TBD        | TBD           | Initial deployment |

Append new rows when producing a new bundle; keep old `.pkl` files for rollback.

## Rollback

1. Set `scratch_classifier_enabled=false` via DB.
2. (Optional) Swap `scratch_bundle_path` to previous `_vN.pkl`, set enabled=true.
