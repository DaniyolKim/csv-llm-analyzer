@echo off
echo CSV 분석기를 실행합니다...

REM 가상 환경이 존재하는지 확인
if not exist venv (
    echo 가상 환경이 존재하지 않습니다. setup_venv.bat를 먼저 실행해주세요.
    pause
    exit
)

echo 가상 환경을 활성화합니다...
call venv\Scripts\activate

echo Streamlit 애플리케이션을 실행합니다...
streamlit run app.py

pause