# LibRaw + rawpy Windows 编译安装指南

## 适用环境

- Windows 10/11
- Conda 环境（Python 3.8）
- Visual Studio 2019 或 2022 BuildTools（C++ 生成工具 + Windows 10 SDK）

---

## 1. 解压源码

```batch
cd downloads
unzip LibRaw-0.21.1.zip
unzip rawpy.zip
```

执行后得到 `LibRaw-0.21.1\` 和 `rawpy\` 两个目录。

---

## 2. 编译 LibRaw（MSBuild）

> LibRaw 0.21.1 不含 CMakeLists.txt，需用自带的 `.sln` / `.vcxproj` 编译。

### 2.1 创建 build_libraw.bat

在 `downloads\` 下新建 `build_libraw.bat`：

```batch
@echo off
setlocal

:: 激活 VS 编译环境
:: VS 2019 BuildTools 默认路径:
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
:: VS 2022 通常是:
:: call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cd /d "%~dp0LibRaw-0.21.1"

:: 自动检测 Windows SDK 版本
for /f "tokens=*" %%i in ('dir /b /ad "C:\Program Files (x86)\Windows Kits\10\Include\10.*" 2^>nul') do set SDK_VER=%%i
if "%SDK_VER%"=="" (
    echo ERROR: Windows SDK not found
    exit /b 1
)

echo Using Windows SDK %SDK_VER%

:: 编译（v142=VS2019, v143=VS2022，根据实际安装的 VS 版本选择）
msbuild LibRaw.sln /t:libraw /p:Configuration=Release /p:Platform=x64 ^
  /p:WindowsTargetPlatformVersion=%SDK_VER% /p:PlatformToolset=v142 /m

if %ERRORLEVEL% NEQ 0 exit /b 1

:: 复制产物
set "INSTALL=%~dp0libraw_install"
if not exist "%INSTALL%\bin" mkdir "%INSTALL%\bin"
if not exist "%INSTALL%\lib" mkdir "%INSTALL%\lib"
if not exist "%INSTALL%\include\libraw" mkdir "%INSTALL%\include\libraw"

copy /Y buildfiles\release-x86_64\libraw.dll "%INSTALL%\bin\"
copy /Y buildfiles\release-x86_64\libraw.lib "%INSTALL%\lib\"
copy /Y libraw\*.h "%INSTALL%\include\libraw\"

echo Build complete.
endlocal
```

### 2.2 MSBuild 参数说明

| 参数 | 作用 |
|------|------|
| `/p:PlatformToolset=v142` | VS 2019 工具集；VS 2022 改用 `v143` |
| `/p:WindowsTargetPlatformVersion=%SDK_VER%` | 自动使用已安装的 SDK 版本 |

### 2.3 执行编译

```batch
build_libraw.bat
```

编译完成后 `libraw_install\` 结构：

```
libraw_install/
├── bin/
│   └── libraw.dll
├── lib/
│   └── libraw.lib
└── include/
    └── libraw/   (7 个 .h)
```

---

## 3. 安装 rawpy

> rawpy 的 `setup.py` 已预先修改，支持 `RAWPY_USE_SYSTEM_LIBRAW` 环境变量。

### 3.1 降级 Cython

```bash
conda activate dark-isp
pip install "cython<3"
```

### 3.2 编译安装

```batch
set RAWPY_USE_SYSTEM_LIBRAW=1
set LIBRAW_PATH=.\libraw_install

cd rawpy
pip install -v -e .
```

### 3.3 复制运行时 DLL

将 `libraw.dll` 加入 DLL 搜索路径（二选一）：

**方案 A**（推荐）：复制到 rawpy 包目录
```batch
copy libraw_install\bin\libraw.dll rawpy\rawpy\
```

**方案 B**：将 `libraw_install\bin` 加入系统 PATH 环境变量

### 3.4 验证

```bash
python -c "import rawpy; print(rawpy.__version__); r = rawpy.RawPy(); print('OK')"
```

输出 `0.16.0` 和 `OK` 即成功。

---

## 常见问题

### Q1: MSBuild 报 "找不到 Windows SDK 版本"

脚本会自动检测已安装的 SDK。如失败，手动查看：
```batch
dir "C:\Program Files (x86)\Windows Kits\10\Include"
```
然后改 `build_libraw.bat` 中的 `/p:WindowsTargetPlatformVersion=你的版本`。

### Q2: MSBuild 报 "无法找到 v142/v143 生成工具"

检查 VS 安装的版本，修改 `build_libraw.bat` 中 `/p:PlatformToolstep` 的值：
- VS 2019 → `v142`
- VS 2022 → `v143`

### Q3: CUDA 编译 mmcv 时报 "unsupported Microsoft Visual Studio version"

VS 版本太新，CUDA 不支持。需安装 VS 2019 BuildTools（v142 工具集），或用 `-allow-unsupported-compiler` 绕过（不保证稳定性）。

### Q4: import rawpy 报 DLL load failed

DLL 未找到。方案 A：复制到 rawpy 包目录；方案 B：加入 PATH。

### Q5: Cython 报 "Assignment to a read-only property"

Cython 版本过高，降级：`pip install "cython<3"`

---

## 备选方案

```bash
conda activate dark-isp
conda install -c conda-forge rawpy
```
