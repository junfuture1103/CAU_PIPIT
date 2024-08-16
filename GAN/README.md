Anaconda 환경에서 requirements.txt 파일을 사용한 환경 구축
Anaconda에서는 pip로 설치된 패키지를 포함하는 환경을 설정할 수 있습니다. requirements.txt 파일을 사용하여 Anaconda 환경을 만드는 방법은 다음과 같습니다:

새로운 Conda 환경 생성:
새로운 Anaconda 환경을 만들 때는 conda create 명령어를 사용합니다. Python 버전도 명시할 수 있습니다.

```
conda create -n myenv python=3.8
```

myenv는 환경의 이름입니다. 원하는 이름으로 바꾸면 됩니다.

환경 활성화:
생성한 환경을 활성화합니다.

```
conda activate myenv
```

requirements.txt 파일을 사용하여 패키지 설치:
requirements.txt 파일에 정의된 패키지를 설치합니다. 이 과정에서 pip 명령어를 사용할 수 있습니다.

```
pip install -r requirements.txt
```

Conda는 conda install 명령어도 제공하지만, requirements.txt 파일에 정의된 패키지는 pip를 통해 설치하는 것이 일반적입니다.

```
#start action
cd scripts
python3 main_script.py
```

환경 비활성화:
작업이 끝난 후 환경을 비활성화할 수 있습니다.

```
conda deactivate
```

이 방법을 사용하면 Anaconda 환경에서 pip로 설치된 패키지까지 포함하여 필요한 모든 패키지를 손쉽게 관리하고 설치할 수 있습니다.