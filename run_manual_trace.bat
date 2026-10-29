@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
start "" "%PYTHON_EXE%" "%APP_DIR%app.py"
endlocal
