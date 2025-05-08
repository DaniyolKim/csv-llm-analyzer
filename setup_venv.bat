@echo off
echo CSV 분석기 가상 환경 설정을 시작합니다...

REM 가상 환경이 이미 존재하는지 확인
if exist venv (
    echo 가상 환경이 이미 존재합니다. 삭제 후 재생성합니다.
    rmdir /s /q venv
)

echo 가상 환경을 생성합니다...
python -m venv venv

echo 가상 환경을 활성화합니다...
call venv\Scripts\activate

echo 필요한 패키지를 설치합니다...
pip install -r requirements.txt

echo.
echo 설정이 완료되었습니다!
echo 애플리케이션을 실행하려면 다음 명령어를 입력하세요:
echo venv\Scripts\activate
echo streamlit run app.py
echo.

pause