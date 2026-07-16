@echo off
REM Watch the current grok /compact -> RLM ingest ("memory cycle") to completion.
REM Double-click this, or run it from any terminal. It refreshes until the cycle is
REM DONE, then beeps and shows a "safe to continue" banner.
chcp 65001 >nul
C:\Python313\python.exe "%~dp0hooks\rlm_watch_ingest.py" %*
echo.
pause
