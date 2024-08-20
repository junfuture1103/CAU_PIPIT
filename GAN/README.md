CAU_PIPIT GAN 실습을 위한 환경세팅 설명문서입니다.
우선 각자 모든 환경이 다르기 때문에, python 가상환경 venv를 이용합니다.

모든과정 촬영영상 (약 5분)
https://youtu.be/vbANYv4F5r8

파이썬 가상환경을 이용하면 파이썬 라이브러리를 본래 컴퓨터에 있는 환경과는 독립적으로
설치 및 관리할 수 있어 매우 용이합니다.

### 1. 가상환경 생성
먼저, 파이썬 가상환경을 생성하기 위해 터미널이나 명령 프롬프트를 엽니다.


python -m venv CAU_PIPIT_VENV
CAU_PIPIT_VENV는 가상환경의 이름입니다. 
원하는 이름으로 변경할 수 있지만

추후 github에 올릴때 이 파일은 제거해야하므로(용량이 너무큼)
이름을 반드시 CAU_PIPIT_VENV로 해줍니다.

### 2. 가상환경 활성화
가상환경을 활성화하려면 아래 명령을 사용하세요.
```
Windows:
myenv\Scripts\activate
```
```
macOS/Linux:
source myenv/bin/activate
```
가상환경이 활성화되면 터미널 프롬프트 앞에 (myenv)와 같은 표시가 나타납니다.

### 3. requirements.txt를 사용하여 모듈 설치
다른 환경에서 동일한 모듈을 설치하려면, 아래 명령을 사용하여 requirements.txt 파일에 명시된 모듈을 일괄 설치할 수 있습니다.
requirements.txt는 이미 사전에 생성해두었기 때문에, 존재하는 requirements.txt로 설치하면됩니다.
```
pip install -r requirements.txt
```
이 명령을 실행하면 requirements.txt에 있는 모든 모듈이 가상환경에 설치됩니다.

### 4. 가상환경 비활성화
작업을 마친 후, 가상환경을 비활성화하려면 아래 명령을 사용하세요.
```
deactivate
```
이 명령을 실행하면 가상환경이 비활성화되고, 기본 파이썬 환경으로 돌아갑니다.

### 5. main_script.py 실행
이후 scripts/main_script.py를 실행하면됩니다.
