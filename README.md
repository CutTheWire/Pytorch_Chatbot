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

## 2. Ubuntu 24.04 LTS 설치
Microsoft Store에서 Ubuntu 24.04 LTS를 검색하여 설치합니다.

Microsoft Store url -> https://www.microsoft.com/store/productId/9NZ3KLHXDJP5?ocid=pdpshare

## 3. Ubuntu 초기 설정
Ubuntu를 처음 실행하면 사용자 이름과 비밀번호를 설정합니다.

## 4. GUI 환경 구성
### 4.1 필요한 패키지 설치
Ubuntu 터미널에서 아래 명령어를 통해 필요한 패키지를 설치합니다.
```bash
sudo apt update
sudo apt install xfce4 xfce4-goodies xorg dbus-x11 x11-xserver-utils
```

### 4.2 `xrdp` 설치 및 설정
`xrdp`를 설치하고 설정합니다.
```bash
sudo apt install xrdp
sudo systemctl enable xrdp
sudo systemctl start xrdp
```

### 4.3 `xrdp` 설정 파일 수정
`xrdp` 설정 파일을 열어 기본 세션을 `xfce`로 설정합니다.
```bash
sudo nano /etc/xrdp/startwm.sh
```
파일의 끝부분에 아래 내용을 추가합니다.
```bash
startxfce4
```
예시 파일 템플릿
```sh
  GNU nano 7.2                                      /etc/xrdp/startwm.sh *
#!/bin/sh
# xrdp X session start script (c) 2015, 2017, 2021 mirabilos
# published under The MirOS Licence

# Rely on /etc/pam.d/xrdp-sesman using pam_env to load both
# /etc/environment and /etc/default/locale to initialise the
# locale and the user environment properly.

if test -r /etc/profile; then
        . /etc/profile
fi

if test -r ~/.profile; then
        . ~/.profile
fi

# 기존의 Xsession 시작 명령을 주석 처리하거나 삭제합니다.
# test -x /etc/X11/Xsession && exec /etc/X11/Xsession
# exec /bin/sh /etc/X11/Xsession

# xfce4 세션을 시작하도록 설정합니다.
startxfce4
```
수정한 후, Ctrl + O를 눌러서 저장하고 Ctrl + X를 눌러서 파일을 닫습니다.

## 5. 원격 데스크탑 연결
### 5.1 WSL2 Ubuntu의 IP 주소 확인
WSL2 Ubuntu 터미널에서 IP 주소를 확인합니다.
```bash
ip addr show eth0 | grep inet
```
```bash
cutthewire@DESKTOP:~$ ip addr show eth0 | grep inet
    inet 172.24.113.222/20 brd 172.24.127.255 scope global eth0
    ...
```
예: `172.24.113.222`

### 5.2 Windows 방화벽 설정 확인
Windows 방화벽에서 원격 데스크탑 연결이 허용되어 있는지 확인합니다.
- 제어판 -> 시스템 및 보안 -> Windows Defender 방화벽 -> 고급 설정 -> 인바운드 규칙에서 `원격 데스크톱 - 사용자 모드(TCP-In)` 규칙이 활성화되어 있는지 확인합니다.

### 5.3 네트워크 연결 테스트
PowerShell에서 WSL2 Ubuntu의 IP 주소로 연결할 수 있는지 테스트합니다.

예: `172.24.113.222`
```powershell
Test-NetConnection -ComputerName 172.24.113.222 -Port 3389                    
```
```powershell
PS C:\Users\cutthewire> Test-NetConnection -ComputerName 172.24.113.222 -Port 3389                                                                                                                                                                                                                                                                                            ComputerName     : 172.24.113.222
RemoteAddress    : 172.24.113.222
RemotePort       : 3389
InterfaceAlias   : vEthernet (WSL (Hyper-V firewall))
SourceAddress    : 172.24.112.1
TcpTestSucceeded : True
```
`TcpTestSucceeded : True`가 나타나면 연결이 성공한 것입니다.

### 5.4 원격 데스크탑 클라이언트를 통한 연결
Windows에서 `원격 데스크톱 연결` 프로그램을 실행하고, WSL2 Ubuntu의 IP 주소를 입력하여 연결합니다.
- 사용자 이름과 비밀번호를 입력하여 로그인합니다.

## 6. 문제 해결
### 6.1 `xrdp` 로그 파일 확인
연결에 문제가 있을 경우 `xrdp` 로그 파일을 확인합니다.
```bash
sudo cat /var/log/xrdp.log
sudo cat /var/log/xrdp-sesman.log
```

### 6.2 `xrdp` 및 `sesman` 서비스 재시작
서비스를 재시작합니다.
```bash
sudo systemctl restart xrdp
sudo systemctl restart xrdp-sesman
```
문제가 생길 경우 종료는 PowerShell 사용
```PowerShell
wsl --shutdown
wsl
```
### 7.1. `.wslconfig` 파일 수정
`.wslconfig` 파일을 열어서 CPU 코어 수를 조정합니다. 예를 들어, 시스템에 총 20개의 코어가 있고 20%를 사용하고자 한다면, 4개의 코어를 할당하면 됩니다. 또한 사용중인 메모리에 맞춰 성능을 제한을 해주세요. (64GB 사용 중이므로 16GB 설정)

```ini
[wsl2]
memory=16GB
processors=4
```

### 7.2. WSL2 재시작
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
- 우분투 24.04 LTS 환경에 맞는 키 링크는 다음과 같습니다:

```bash
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
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
