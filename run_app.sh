#!/bin/bash

echo "CSV 분석기를 실행합니다..."

# 가상 환경이 존재하는지 확인
if [ ! -d "venv" ]; then
    echo "가상 환경이 존재하지 않습니다. setup_venv.sh를 먼저 실행해주세요."
    exit 1
fi

echo "가상 환경을 활성화합니다..."
source venv/bin/activate

echo "Streamlit 애플리케이션을 실행합니다..."
streamlit run app.py