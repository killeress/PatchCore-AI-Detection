CAPI PaddleOCR MARK Shadow — Offline Production Install
========================================================

Target:
  RHEL 9 compatible Linux, x86_64, CPU, no external network.

Default locations:
  CAPI application : /root/Code/CAPI_AD
  Shadow runtime   : /aidata/capi_ai/mark_shadow
  Shadow database  : /aidata/capi_ai/mark_shadow/data/mark_shadow.db

Install and activate:
  cd <extracted bundle>
  sudo ./install.sh

Install without restarting the main CAPI server:
  sudo ./install.sh --no-restart-capi

Override a non-standard CAPI directory:
  sudo CAPI_APP_ROOT=/path/to/CAPI_AD ./install.sh

Verify:
  sudo ./scripts/verify_offline.sh

Health and statistics:
  curl http://127.0.0.1:8765/health
  curl http://127.0.0.1:8765/stats

Admin comparison page:
  Sign in to /settings as an administrator and open "Mark PPOCR檢查".

The installer:
  1. verifies every bundled file;
  2. installs an isolated Python/PaddleOCR runtime and local models;
  3. installs and starts capi-mark-shadow.service;
  4. backs up server_config.yaml;
  5. enables the non-blocking mark_shadow client;
  6. restarts the main CAPI service unless --no-restart-capi is used.

The PaddleOCR result is the primary formal MARK text used by AOI and QJPG.
The existing DotMatrixCV logic still locates the crop and exclusion region,
and is used as a fallback only when PaddleOCR is unavailable or returns no
valid two-character result.
