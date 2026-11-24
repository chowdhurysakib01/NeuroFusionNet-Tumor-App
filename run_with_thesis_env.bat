@echo off
cd /d "D:\NeuroFusionNet Tumor App"
if not exist "D:\Thesis\.venv\Scripts\python.exe" (
  echo Could not find D:\Thesis\.venv\Scripts\python.exe
  pause
  exit /b 1
)
"D:\Thesis\.venv\Scripts\python.exe" -m streamlit run app.py
pause
