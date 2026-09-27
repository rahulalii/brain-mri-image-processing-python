@echo off
REM ========================================================
REM  Watershed Pipeline – One-click Setup & Run
REM ========================================================
echo.
echo  Searching for Python ...
echo.

REM Try common locations
SET PY=
IF EXIST "C:\Python312\python.exe"  SET PY=C:\Python312\python.exe
IF EXIST "C:\Python311\python.exe"  SET PY=C:\Python311\python.exe
IF EXIST "C:\Python310\python.exe"  SET PY=C:\Python310\python.exe
IF EXIST "C:\Python39\python.exe"   SET PY=C:\Python39\python.exe
IF EXIST "C:\Python38\python.exe"   SET PY=C:\Python38\python.exe

REM Try the user's local AppData Python installs
FOR /D %%G IN ("%LOCALAPPDATA%\Programs\Python\Python3*") DO SET PY=%%G\python.exe

REM Try Anaconda / Miniconda
IF EXIST "%USERPROFILE%\anaconda3\python.exe"  SET PY=%USERPROFILE%\anaconda3\python.exe
IF EXIST "%USERPROFILE%\miniconda3\python.exe" SET PY=%USERPROFILE%\miniconda3\python.exe

IF "%PY%"=="" (
    echo  [ERROR] Python not found automatically.
    echo  Please install Python from https://www.python.org/downloads/
    echo  and make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  Found Python: %PY%
echo.
echo  Installing / upgrading dependencies ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt

echo.
echo  Running the Watershed Pipeline ...
"%PY%" watershed_pipeline.py

pause
