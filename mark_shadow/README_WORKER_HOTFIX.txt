CAPI MARK PaddleOCR Worker Update 2026.07.31.1
================================================

Changes:
  Convert grayscale or BGRA MARK crops to 3-channel BGR before PaddleOCR.
  This fixes per-row error "tuple index out of range".
  Save both agreed and disagreed crops for the admin comparison page.
  Return PaddleOCR engine, model, and worker API versions for formal logs.
  Worker API version is now v2.

Install on the production host:
  cd <extracted hotfix directory>
  sudo ./scripts/install_worker_hotfix.sh

Verify:
  curl http://127.0.0.1:8765/health

Existing error rows are preserved. After a new panel is collected, verify that
the newest row has latency_ms greater than zero, an empty error value, and a
non-empty crop_path.
