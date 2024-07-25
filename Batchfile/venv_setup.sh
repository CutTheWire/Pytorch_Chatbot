#!/bin/bash

# UTF-8 설정
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# 가상 환경 디렉토리 이름 설정
ENV_DIR=".venv"

# Python 설치 경로 설정 (Python 3.11.2)
PYTHON_PATH="/usr/bin/python3.11"

# Python 3.11.2가 설치되어 있는지 확인
if ! [ -x "$(command -v $PYTHON_PATH)" ]; then
  echo "Python 3.11.2이(가) 설치되어 있지 않습니다. 설치를 진행합니다."
  sudo apt update
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv
fi

# 가상 환경 생성
$PYTHON_PATH -m venv $ENV_DIR

echo "가상 환경 활성화 중..."
source $ENV_DIR/bin/activate

echo "가상 환경이 성공적으로 생성되고 활성화되었습니다."
echo "가상 환경을 활성화하려면 다음 명령을 사용하세요:"
echo "source $ENV_DIR/bin/activate"
