@echo off
cd /d "%~dp0"
if not exist frontend\node_modules (
  echo Installing UI packages ^(one time^)...
  pushd frontend
  call npm install --no-audit --no-fund
  popd
)
echo Building the React UI...
pushd frontend
call npm run build
popd
start "" http://127.0.0.1:8000
python -m uvicorn app:app --port 8000 --host 127.0.0.1
