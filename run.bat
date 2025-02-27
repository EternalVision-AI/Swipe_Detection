@echo off
start "" python websocket_server.py
start "" python cvmain.py
start "" ./SwipeGame_Web/index.html
pause
