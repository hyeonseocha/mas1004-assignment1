@echo off
cd /d "C:\Users\sk408\Documents\mas1004-assignment1"
echo Cleaning partial results...
if exist "data\raw\reusable_clear_cups" del /q "data\raw\reusable_clear_cups\*.jpg" 2>nul
if exist "data\raw\disposable_plastic_takeout_cups" del /q "data\raw\disposable_plastic_takeout_cups\*.jpg" 2>nul
if exist "data\raw\ceramic_cafe_mugs" del /q "data\raw\ceramic_cafe_mugs\*.jpg" 2>nul
echo Starting download...
.venv\Scripts\python.exe src\collect.py --classes "Reusable clear cups,Disposable plastic takeout cups,Ceramic cafe mugs" --n 150 1>collect_output.txt 2>&1
echo Done! >> collect_output.txt