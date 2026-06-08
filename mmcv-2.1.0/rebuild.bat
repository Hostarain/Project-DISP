@echo off
set DISTUTILS_USE_SDK=1
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\mmcv-2.1.0"
D:\conda\envs\dark-isp\python.exe setup.py develop
