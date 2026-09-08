CAPI MARK PaddleOCR Worker Update (2026.09.08.3)
==============================================

Changes:
  Apply the administrator-managed character conflict rules received from CAPI
  before temporal stabilization. Each matching pair can specify an output
  character, including equal inputs (0 + 0 -> O). Matched positions also take
  priority over the final temporal result. The default U/V rule remains enabled until an
  administrator edits or removes it in Mark PPOCR inspection settings.
  Preserve raw PaddleOCR results and record the applied rule and positions.
  Worker API version is now v5. Update both CAPI and this worker for the new
  output selector; older workers may reject equal-input rules.

For a MARK that must use letter O when DotMatrixCV reads digit 0, save both:
  Paddle 0 + DotMatrixCV 0 -> output O
  Paddle O + DotMatrixCV 0 -> output O
Rules apply to all models and either character position. They also convert a
real digit 0 when the pair matches. Existing rules without an output field
continue to use the DotMatrixCV character. Raw OCR and old result rows remain
available; new decisions reload history using the current conversion rules.

Install on the production host:
  cd <extracted hotfix directory>
  sudo ./scripts/install_worker_hotfix.sh

Verify:
  curl http://127.0.0.1:8765/health

Existing error rows are preserved. After a new panel is collected, verify that
the newest row has worker_version 5 and forced_char_conversion in
adoption_reason when a configured conflict occurs.
