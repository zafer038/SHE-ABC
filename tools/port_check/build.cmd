@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS harness.cpp cec17_test_func.cpp /Fe:cec17.exe
