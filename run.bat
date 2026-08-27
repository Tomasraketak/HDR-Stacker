@echo off
REM ---------------------------------------------------------------
REM  Astro HDR Stacker - spousteci skript pro Windows 10 / 11
REM  Pri prvnim spusteni vytvori virtualni prostredi a nainstaluje
REM  knihovny. Pri dalsich spustenich uz jen spusti aplikaci.
REM ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Vytvarim virtualni prostredi .venv ...
    %PY% -m venv .venv
    if errorlevel 1 goto :nopython
    echo [2/2] Instaluji knihovny ^(muze trvat par minut^) ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :pipfail
)

echo Spoustim Astro HDR Stacker ...
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto :runfail
goto :eof

:nopython
echo.
echo CHYBA: Nenasel jsem Python. Nainstalujte Python 3.10 az 3.12 z python.org
echo a pri instalaci zaskrtnete "Add Python to PATH".
pause
exit /b 1

:pipfail
echo.
echo CHYBA: Instalace knihoven selhala. Zkontrolujte pripojeni k internetu.
pause
exit /b 1

:runfail
echo.
echo Aplikace skoncila s chybou. Vypis vyse muze napovedet proc.
pause
exit /b 1
