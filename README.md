# Vulkan AI Samples - MNIST Neural Network

Vulkan을 사용한 GPU 가속 MNIST 손글씨 숫자 인식 신경망 구현

## 프로젝트 구조

- **10-mnist/** - 초기 Vulkan 기반 MNIST 구현
- **11-mnist-refactor/** - 리팩토링된 버전 (템플릿 코드를 일반 구현으로 분리)
- **external/** - 외부 의존성 라이브러리

## 주요 기술 스택

- **C++23**
- **Vulkan** - GPU 컴퓨트 가속
- **SPIRV** - 셰이더 컴파일 및 리플렉션
- **GLFW** - 윈도우 관리
- **nlohmann/json** - JSON 파싱 (가중치 로드)

## 필요한 의존성

### 시스템 요구사항

1. **Vulkan SDK** (버전 1.3 이상)
   ```bash
   # Ubuntu/Debian
   sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers

   # 또는 LunarG SDK 다운로드
   # https://vulkan.lunarg.com/sdk/home
   ```

2. **CMake** (버전 3.16 이상)
   ```bash
   sudo apt install cmake
   ```

3. **C++ 컴파일러** (C++23 지원)
   ```bash
   # GCC 13 이상 또는 Clang 16 이상
   sudo apt install g++-13
   ```

4. **윈도우 시스템 라이브러리** (GLFW용)
   ```bash
   # Wayland 지원 (권장)
   sudo apt install libwayland-dev wayland-protocols libxkbcommon-dev

   # 또는 X11만 사용
   sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
   ```

5. **기타 개발 도구**
   ```bash
   sudo apt install build-essential git
   ```

### 의존성 확인

```bash
# Vulkan 설치 확인
vulkaninfo --summary

# CMake 버전 확인
cmake --version

# C++ 컴파일러 확인
g++ --version
```

## 빌드 방법

### 1. 저장소 클론 및 서브모듈 초기화

```bash
git clone <repository-url>
cd vai-samples

# 중요: 중첩된 서브모듈까지 모두 초기화해야 합니다
git submodule update --init --recursive
```

**주의**: SPIRV-Tools는 내부적으로 SPIRV-Headers를 서브모듈로 사용합니다.
`--recursive` 옵션을 사용하지 않으면 CMake 설정 시 다음 에러가 발생합니다:
```
CMake Error: SPIRV-Headers was not found - please checkout a copy at external/spirv-headers.
```

이미 클론한 경우 다음 명령으로 서브모듈을 초기화하세요:
```bash
cd /home/hwoo-joo/github/vai-samples
git submodule update --init --recursive

# 대소문자 문제 해결을 위한 심볼릭 링크 생성
cd external
ln -s SPIRV-Headers spirv-headers
cd ..
```

### 2. 필수 시스템 패키지 설치

빌드 전에 모든 필요한 패키지를 설치하세요:

```bash
# 모든 필수 의존성 한번에 설치
sudo apt install \
  vulkan-tools libvulkan-dev vulkan-validationlayers \
  cmake build-essential git g++-13 \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libwayland-dev wayland-protocols libxkbcommon-dev
```

### 3. 빌드 디렉토리 생성 및 CMake 설정

```bash
# 빌드 디렉토리 생성
mkdir build
cd build

# Release 모드로 CMake 설정
cmake .. -DCMAKE_BUILD_TYPE=Release

# 또는 Debug 모드
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### 4. 컴파일

```bash
# 빌드 실행 (멀티코어 사용)
cmake --build . -j$(nproc)

# 또는 make 직접 사용
make -j$(nproc)
```

### 5. 빌드 출력 위치

- **Release 빌드**: `bin/release/`
- **Debug 빌드**: `bin/debug/`

## 실행 방법

### 중요: 실행 전 준비사항

프로그램은 `PROJECT_ROOT_DIR/data/*.png` 경로에서 이미지를 찾습니다. 먼저 data 디렉토리를 만들고 이미지를 복사하세요:

```bash
cd /home/hwoo-joo/github/vai-samples
mkdir -p data
cp 10-mnist/*.png data/
```

### 10-mnist 실행

```bash
# 프로젝트 루트 디렉토리에서 실행해야 합니다
cd /home/hwoo-joo/github/vai-samples
./bin/release/10-mnist
```

**예상 출력**:
```
Found 3 suitable physical devices:
[GPU 0] Device Name: NVIDIA GeForce RTX 5090 ...
...
[MNIST evaluation: 1000 iterations] => 839ms
data[0] = 23.026684   <- 가장 높은 값, 숫자 0으로 인식
data[1] = -24.376102
...
```

### 11-mnist-refactor 실행

```bash
cd /home/hwoo-joo/github/vai-samples
./bin/release/11-mnist-refactor
```

### 테스트 이미지

프로젝트에는 샘플 MNIST 이미지가 포함되어 있습니다:
- `0.png`, `1.png`, `4.png`, `5.png`, `9.png`

출력값 배열에서 가장 높은 값의 인덱스가 인식된 숫자입니다.

## 프로젝트 주요 구성 요소

### 10-mnist/

| 파일 | 설명 |
|------|------|
| `main.cpp` | 프로그램 진입점 |
| `vulkanApp.h/cpp` | Vulkan 초기화 및 컴퓨트 파이프라인 관리 |
| `neuralNet.h` | 신경망 구조 정의 |
| `neuralNodes.h` | 신경망 레이어 구현 (Dense, ReLU, Softmax 등) |
| `ndarray.h` | N차원 배열 (다차원 데이터 구조) |
| `tensor.h` | 텐서 연산 |
| `jsonParser.h/cpp` | JSON 파싱 (가중치 로드) |
| `weights.json` | 학습된 신경망 가중치 (2.1MB) |
| `spirvHelpers.cpp` | SPIRV 바이너리 처리 |

### 11-mnist-refactor/

10-mnist의 개선 버전으로, 다음 변경사항이 있습니다:
- `neuralNodes.cpp` - 템플릿 헤더에서 일반 구현 파일로 분리
- `PythonMNIST.py` - Python으로 신경망 학습 스크립트 추가
- 코드 구조 개선 및 모듈화

## 백엔드 옵션 (CMake)

CMake 설정 시 다양한 백엔드를 선택할 수 있습니다:

```bash
# Vulkan 백엔드 (기본값)
cmake .. -DREQUIRE_BACKEND_VULKAN=ON

# OpenCL 백엔드
cmake .. -DREQUIRE_BACKEND_OPENCL=ON

# CUDA 백엔드
cmake .. -DREQUIRE_BACKEND_CUDA=ON

# Kompute 라이브러리 사용
cmake .. -DUSE_KOMPUTE=ON
```

## 빌드 성공 확인

빌드가 성공하면 다음 위치에 실행 파일이 생성됩니다:

```bash
ls -lh bin/release/
# 출력:
# 10-mnist           - MNIST 신경망 실행 파일 (약 7.9MB)
# 11-mnist-refactor  - 리팩토링된 버전 (약 7.9MB)
# spirv-* 도구들     - SPIRV 관련 유틸리티들
```

## 문제 해결

### 컴파일 에러 (리눅스/Clang 환경)

이 프로젝트는 원래 Windows/MSVC 환경용으로 작성되었습니다. 리눅스에서 빌드하려면 다음 수정사항들이 필요합니다:

**이미 수정된 항목** (이 README를 따라 진행했다면 자동 적용됨):
- ✅ `<cstring>` 헤더 추가 (strcmp, memcpy 사용을 위해)
- ✅ `error.h`에 `_ASSERT` 매크로 정의
- ✅ `Edge` 클래스 전방 선언
- ✅ 람다 표현식에 명시적 타입 지정
- ✅ `assert_impl` 함수를 템플릿으로 변경 (shared_ptr 지원)

**경고 메시지** (무시 가능):
- `[-Wformat]` - printf 포맷 경고
- `[-Wformat-security]` - printf 보안 경고
- `[-Wreturn-type]` - 리턴값 경로 경고

이러한 경고들은 프로그램 실행에 영향을 주지 않습니다.

### SPIRV-Headers 누락 에러

**에러 메시지**:
```
CMake Error at external/SPIRV-Tools/external/CMakeLists.txt:87 (message):
  SPIRV-Headers was not found - please checkout a copy at external/spirv-headers.
```

**해결 방법**:
```bash
# 프로젝트 루트 디렉토리에서
cd /home/hwoo-joo/github/vai-samples/external

# 소문자 이름의 심볼릭 링크 생성
ln -s SPIRV-Headers spirv-headers

# 빌드 디렉토리로 돌아가서 CMake 재실행
cd ..
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**원인**: CMakeLists.txt가 `external/spirv-headers` (소문자)를 찾지만, 실제 디렉토리 이름은 `external/SPIRV-Headers` (대문자)입니다. 이는 대소문자를 구분하는 리눅스 파일시스템에서 문제가 됩니다.

### wayland-scanner 누락 에러

**에러 메시지**:
```
CMake Error at external/glfw/src/CMakeLists.txt:77 (message):
  Failed to find wayland-scanner
```

**해결 방법 1** (권장): Wayland 개발 패키지 설치
```bash
# Ubuntu/Debian
sudo apt install libwayland-dev wayland-protocols libxkbcommon-dev

# 그 후 CMake 재실행
cd /home/hwoo-joo/github/vai-samples/build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**해결 방법 2**: X11만 사용하도록 GLFW 설정
```bash
# X11 개발 패키지만 설치 (이미 설치되어 있을 가능성 높음)
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# GLFW가 Wayland를 사용하지 않도록 설정
cd /home/hwoo-joo/github/vai-samples/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGLFW_BUILD_WAYLAND=OFF
```

**원인**: GLFW는 리눅스에서 X11과 Wayland 두 가지 윈도우 시스템을 지원합니다. 기본적으로 둘 다 빌드하려고 하는데, Wayland 개발 패키지가 없으면 에러가 발생합니다.

### X11 라이브러리 누락 에러 (RandR, Xinerama 등)

**에러 메시지**:
```
CMake Error at external/glfw/src/CMakeLists.txt:181 (message):
  RandR headers not found; install libxrandr development package
```

또는 유사한 에러:
```
Xinerama headers not found
Xcursor headers not found
Xi headers not found
```

**해결 방법**:
```bash
# 모든 필요한 X11 개발 패키지 한번에 설치
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# CMake 재실행
cd /home/hwoo-joo/github/vai-samples/build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**원인**: GLFW가 X11 윈도우 시스템을 지원하기 위해 필요한 개발 헤더 파일들이 설치되어 있지 않습니다.

### Vulkan을 찾을 수 없는 경우

```bash
# Vulkan SDK 경로 수동 지정
cmake .. -DVULKAN_SDK=/path/to/vulkan/sdk
```

### C++23 지원 에러

```bash
# 컴파일러 명시적 지정
cmake .. -DCMAKE_CXX_COMPILER=g++-13
```

### 서브모듈 일반 문제

```bash
# 서브모듈 강제 업데이트
git submodule update --init --recursive --force

# 서브모듈 상태 확인
git submodule status

# 특정 서브모듈만 업데이트
cd external/SPIRV-Tools
git submodule update --init --recursive
cd ../..
```

### 빌드 정리

```bash
# 빌드 디렉토리 완전 삭제 후 재빌드
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 신경망 아키텍처

이 프로젝트의 MNIST 신경망은 다음과 같은 구조를 가집니다:

1. **입력 레이어**: 28x28 = 784 픽셀
2. **히든 레이어**: Dense + ReLU 활성화 함수
3. **출력 레이어**: 10개 클래스 (0-9 숫자) + Softmax

모든 행렬 연산은 Vulkan 컴퓨트 셰이더를 통해 GPU에서 병렬 처리됩니다.

## 라이선스

이 프로젝트의 라이선스 정보는 저장소의 LICENSE 파일을 참조하세요.

## 참고 자료

- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Compute Shader](https://www.khronos.org/opengl/wiki/Compute_Shader)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
