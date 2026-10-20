@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cl /nologo /O2 /EHsc /LD /D_CRT_SECURE_NO_WARNINGS cec14_dll.cpp cec14_test_func.cpp /Fe:cec14.dll
