CAPI MARK PaddleOCR Worker Update 2026.08.25.1
================================================

Changes:
  When PaddleOCR reads U and DotMatrixCV reads V at the same character
  position, use V as the effective result before temporal stabilization.
  Preserve the raw PaddleOCR result and record dotmatrix_uv_rescue positions.
  Other character conflicts and the reverse V-to-U direction are unchanged.
  Worker API version is now v3.

Install on the production host:
  cd <extracted hotfix directory>
  sudo ./scripts/install_worker_hotfix.sh

Verify:
  curl http://127.0.0.1:8765/health

Existing error rows are preserved. After a new panel is collected, verify that
the newest row has worker_version 3 and dotmatrix_uv_rescue in adoption_reason
when a Paddle U / DotMatrixCV V conflict occurs.
