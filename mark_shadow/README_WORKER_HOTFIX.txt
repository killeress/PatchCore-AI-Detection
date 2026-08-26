CAPI MARK PaddleOCR Worker Update (Unreleased)
==============================================

Changes:
  Apply the administrator-managed character conflict rules received from CAPI
  before temporal stabilization. The default U/V rule remains enabled until an
  administrator edits or removes it in Mark PPOCR inspection settings.
  Preserve raw PaddleOCR results and record the applied rule and positions.
  Worker API version is now v4.

Install on the production host:
  cd <extracted hotfix directory>
  sudo ./scripts/install_worker_hotfix.sh

Verify:
  curl http://127.0.0.1:8765/health

Existing error rows are preserved. After a new panel is collected, verify that
the newest row has worker_version 4 and forced_char_conversion in
adoption_reason when a configured conflict occurs.
