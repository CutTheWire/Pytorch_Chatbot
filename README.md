# APT 설치 및 버전
## CUDA

- Version : 11.8
- Download : [CUDA Toolkit 11.8 Downloads](https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe)

## cuDNN

- Version : 8.7.0
- Download : [Local Installers for Windows](https://developer.nvidia.com/downloads/c118-cudnn-windows-8664-87084cuda11-archivezip)
- cuDNN directory location
    ```
    C:\tools\cuda\
    ```

## Python

- Version : 3.11.x
- Download : [Python 3.11.4 - June 6, 2023](https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe)

## PyTorch

- Run this Commandpip

    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## 환경 변수 설정

| 변수 이름 | 변수 값 |
| --- | --- |
| CUDA_HOME | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 |

| 변수 이름 | 변수 값 |
| --- | --- |
| CUDNN_HOME | C:\tools\cuda |

| Set | | Path |
| --- | --- | --- |
|SET PATH |=|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%|
|SET PATH |=|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64;%PATH%|
|SET PATH |=|C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include;%PATH%|
|SET PATH |=|C:\tools\cuda\bin;%PATH%|

아래는 WSL2를 설정하고 Microsoft Store에서 Ubuntu 24.04 LTS를 설치하여 GUI 환경을 구성하고 원격 데스크탑을 통해 연결하는 방법을 마크다운 형식으로 정리한 내용입니다.

<br><br>
---

# WSL2와 Ubuntu 24.04 LTS를 사용한 GUI 환경 구성 및 원격 데스크탑 연결


## 1. WSL2 설치 및 설정
<br>

### 1.1 WSL2 활성화
PowerShell을 관리자 권한으로 실행하고 아래 명령어를 입력하여 WSL과 WSL2를 활성화합니다.
```powershell
wsl --install
```

### 1.2 기본 WSL 버전을 WSL2로 설정
기본 WSL 버전을 WSL2로 설정합니다.
```powershell
wsl --set-default-version 2
```

## 2.2 Ubuntu 24.04 LTS 설치
Microsoft Store에서 Ubuntu 22.04 LTS를 검색하여 설치합니다.

Microsoft Store url -> https://www.microsoft.com/store/productId/9PN20MSR04DW?ocid=pdpshare

## 2.3 VSCode에 Remote - WSL 확장 설치
1. Windows에서 VSCode를 설치합니다.
2. VSCode를 실행하고, 확장(Extensions) 탭에서 "Remote - WSL"을 검색하여 설치합니다.

## 2.4 WSL 환경에서 VSCode 실행
1. Windows 탐색기에서 작업하고자 하는 폴더를 열어둡니다.
2. VSCode를 실행하고, 왼쪽 하단의 "Remote" 아이콘을 클릭합니다.
3. "Remote-WSL: New Window" 옵션을 선택합니다.
4. 설치된 Ubuntu 22.04 LTS 배포판을 선택합니다.

## 2.5 WSL 환경 설정
1. VSCode가 WSL 환경에서 실행되면, 필요한 확장 프로그램을 설치할 수 있습니다.
2. 터미널에서 필요한 개발 도구를 설치할 수 있습니다. 예:
   ```
   sudo apt update
   sudo apt install build-essential
   ```

## 2.6 프로젝트 시작
이제 WSL 환경에서 VSCode를 사용하여 프로젝트를 시작할 수 있습니다.

## 2.7 문제 해결
- WSL 연결에 문제가 있을 경우, Windows PowerShell에서 다음 명령어로 WSL을 재시작합니다:
  ```
  wsl --shutdown
  wsl
  ```

### 3.1 `.wslconfig` 파일 수정
`.wslconfig` 파일을 열어서 CPU 코어 수를 조정합니다. 예를 들어, 시스템에 총 20개의 코어가 있고 20%를 사용하고자 한다면, 4개의 코어를 할당하면 됩니다. 또한 사용중인 메모리에 맞춰 성능을 제한을 해주세요. (64GB 사용 중이므로 16GB 설정)

```ini
[wsl2]
memory=16GB
processors=4
```

### 3.2 WSL2 재시작
변경 사항을 적용하기 위해 WSL2 인스턴스를 재시작합니다. PowerShell을 열고 다음 명령을 실행합니다:

```powershell
wsl --shutdown
```

그런 다음 WSL2 인스턴스를 다시 시작합니다:

```powershell
wsl
```

이제 WSL2 인스턴스는 설정한 메모리와 CPU 코어 수를 사용할 것입니다. 다시 한 번 확인하려면 다음 명령어들을 실행합니다:

```bash
free -h
nproc
```
```bash
cutthewire@DESKTOP:~$ free -h
nproc
               total        used        free      shared  buff/cache   available
Mem:            15Gi       559Mi        14Gi       3.8Mi       712Mi        15Gi
Swap:             0B          0B          0B
4
```
이 명령어들을 통해 할당된 메모리와 CPU 코어 수를 확인할 수 있습니다. 


---
이제 WSL2와 Ubuntu 24.04 LTS를 사용하여 GUI 환경을 구성하고 원격 데스크탑을 통해 연결할 수 있습니다.

이 가이드를 따라하면 WSL2와 Ubuntu 24.04 LTS를 통해 GUI 환경을 구성하고 원격 데스크탑을 통해 연결할 수 있습니다. 문제가 발생할 경우 로그 파일을 확인하고 서비스 재시작을 시도해보세요. 

[Learn Microsoft - WSL으로 Linux GUI 앱 실행](https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/gui-apps)

[Microsoft Apps - Ubuntu 24.04 LTS - Windows에서 무료 다운로드 및 설치](https://apps.microsoft.com/detail/9nz3klhxdjp5?hl=ko-kr&gl=KR)

[Learn Microsoft - 이전 버전 WSL의 수동 설치 단계](https://learn.microsoft.com/ko-kr/windows/wsl/install-manual) 

<br><br>

---

# ubuntu 24.04 기준 cuda 설치

## 1. NVIDIA CUDA 공식 설치 가이드 참고
- NVIDIA에서 제공하는 [CUDA 설치 가이드](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)를 참고하여 NVIDIA 저장소 키를 추가해 보겠습니다.
- 우분투 22.04 LTS 환경에 맞는 키 링크는 다음과 같습니다:

```bash
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
```

## 2. CUDA 패키지 설치 문제 해결
- 저장소 키를 해결했다면, 이제 CUDA 패키지를 설치해 보세요:

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```
## 3. CUDA 확인
### CUDA 버전 확인
터미널에서 다음 명령어를 실행하여 설치된 CUDA 버전을 확인할 수 있습니다:
```bash
nvcc --version
```
이 명령어를 실행하면 설치된 CUDA 버전이 출력됩니다.

### GPU 사용 상태 확인

터미널에서 다음 명령어를 실행하여 GPU 사용 상태를 확인할 수 있습니다:
```bash
nvidia-smi
```
이 명령어를 실행하면 GPU 정보와 현재 사용 상태가 출력됩니다.