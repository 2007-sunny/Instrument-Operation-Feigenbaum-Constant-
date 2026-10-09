@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   示波器 / 函數產生器 控制台 — 安裝
echo ============================================
echo.

rem ---- 1. 找 Python ----
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [錯誤] 找不到 Python。
    echo   請先到 https://www.python.org/downloads/ 安裝 Python 3.10 ~ 3.13，
    echo   安裝畫面最下面記得勾選「Add python.exe to PATH」，裝完再重新執行本檔。
    echo.
    pause
    exit /b 1
)
%PY% -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] else 1)"
if errorlevel 1 (
    echo [錯誤] Python 版本太舊，請安裝 Python 3.10 ~ 3.13。
    %PY% --version
    pause
    exit /b 1
)
echo [1/3] 使用的 Python：
%PY% --version
echo.

rem ---- 2. 建立虛擬環境 ----
if exist ".venv\Scripts\python.exe" (
    echo [2/3] 已有虛擬環境 .venv，略過建立。
) else (
    echo [2/3] 建立虛擬環境 .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [錯誤] 建立虛擬環境失敗。
        pause
        exit /b 1
    )
)
echo.

rem ---- 3. 安裝套件（先用官方 PyPI，失敗再用清華鏡像站）----
echo [3/3] 安裝套件（約 100 MB，需要網路，可能要幾分鐘）...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo 官方來源安裝失敗，改用清華大學鏡像站再試一次 ...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
".venv\Scripts\python.exe" -c "import serial, numpy, PyQt5, pyqtgraph" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [錯誤] 套件沒有裝好，請把上面的訊息截圖給會用的人看。
    pause
    exit /b 1
)

echo.
echo ============================================
echo   安裝完成！
echo   - 先雙擊 start_sim.bat 用模擬模式試試看（不用接儀器）
echo   - 接上儀器後雙擊 start.bat
echo ============================================
pause
