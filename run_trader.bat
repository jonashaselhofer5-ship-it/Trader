@echo off
REM Daily trader run — called by Windows Task Scheduler.
cd /d "C:\Users\JonasHaselhoferZARHo\Documents\Claude\Trader"
"C:\Users\JonasHaselhoferZARHo\Documents\Python\envs\trader\python.exe" runner.py >> "logs\scheduler.log" 2>&1
