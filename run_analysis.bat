@echo off
cd /d "C:\Users\BERNARF\OneDrive - National Bank of Belgium\Documents\Veo"
"C:\Users\BERNARF\AppData\Local\Programs\Python\Python312\python.exe" main.py "C:\Users\BERNARF\OneDrive - National Bank of Belgium\Documents\Veo\data\match.mp4" --stride 5 --conf-ball 0.08 > output.log 2>&1
