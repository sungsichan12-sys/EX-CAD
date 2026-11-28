@echo off
setlocal
set "SRC=%~dp0excad_addon"
set "DST=%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\excad_addon"
if defined BLENDER_USER_SCRIPTS set "DST=%BLENDER_USER_SCRIPTS%\addons\excad_addon"
if not exist "%DST%\" mkdir "%DST%"
xcopy /E /I /Y /Q "%SRC%\*.py" "%DST%\" >nul
if errorlevel 1 (
  echo INSTALL FAILED
  exit /b 1
)
echo INSTALLED: %DST%
