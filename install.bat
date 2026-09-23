@echo off
echo Installing EmbedEval AI dependencies...
pip install -r requirements.txt
pushd frontend
npm install --no-audit --no-fund
popd
echo.
echo Done. Run start.bat to launch the API and web application.
