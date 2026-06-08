@echo off
setlocal
call "E:\!yan\Visual Studio\vs\VC\Auxiliary\Build\vcvars64.bat"
cd /d "E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\downloads\LibRaw-0.21.1"

echo ===== Building LibRaw DLL with MSBuild =====
msbuild LibRaw.sln /t:libraw /p:Configuration=Release /p:Platform=x64 /p:WindowsTargetPlatformVersion=10.0.22621.0 /p:PlatformToolset=v143 /m

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: MSBuild failed
    exit /b 1
)

echo ===== Copying output files to install directory =====
set INSTALL=E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\downloads\libraw_install
if not exist "%INSTALL%\bin" mkdir "%INSTALL%\bin"
if not exist "%INSTALL%\lib" mkdir "%INSTALL%\lib"
if not exist "%INSTALL%\include\libraw" mkdir "%INSTALL%\include\libraw"

copy /Y buildfiles\release-x86_64\libraw.dll "%INSTALL%\bin\"
copy /Y buildfiles\release-x86_64\libraw.lib "%INSTALL%\lib\"
copy /Y libraw\*.h "%INSTALL%\include\libraw\"

echo ===== Build complete =====
dir "%INSTALL%\bin\libraw.dll"
dir "%INSTALL%\lib\libraw.lib"
endlocal
