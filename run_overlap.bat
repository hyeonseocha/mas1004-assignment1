@echo off
cd /d "C:\Users\sk408\Documents\mas1004-assignment1"
.venv\Scripts\python.exe src\clean.py overlap > results\overlap.txt 2>&1
echo EXIT CODE: %ERRORLEVEL% >> results\overlap.txt