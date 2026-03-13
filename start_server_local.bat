@echo off
setlocal

:: ============================================================
:: CAPI AI 推論伺服器啟動腳本 (Windows 本地測試)
:: ============================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: 預設使用的是本地設定檔
set CONFIG_FILE=server_config_local.yaml

echo ============================================================
echo   CAPI AI Inference Server (Local Windows)
echo ============================================================
echo   Working dir : %SCRIPT_DIR%
echo   Config      : %CONFIG_FILE%
echo ============================================================

:: 檢查設定檔
if not exist "%CONFIG_FILE%" (
    echo ERROR: Config file not found: %CONFIG_FILE%
    echo Please copy server_config_local.yaml if it doesn't exist.
    pause
    exit /b 1
)

:: 檢查 Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

:: 啟動伺服器 (包含 Web UI)
echo Starting server...
echo Access Web UI at http://localhost:8080 (or the port in config)
echo.

python capi_server.py --config "%CONFIG_FILE%"

pause
