# Official benchmark code

The official benchmark implementations are not redistributed here. Download them and place them as follows.

| Folder | Source | Files |
|---|---|---|
| `external/cec2022-python/Python/` | https://github.com/P-N-Suganthan/2022-SO-BO (Python code) | `CEC2022.py`, `input_data/` |
| `external/cec2017/` | https://github.com/P-N-Suganthan/CEC2017-BoundContrained (`CEC17_fast_pow-C++.zip`) | `cec17_test_func.cpp`, `input_data/` |
| `external/cec2014/` | https://github.com/P-N-Suganthan/CEC2014 (`cec14-c-code.zip`) | `cec14_test_func.cpp`, `input_data/` |

The CEC 2017 runs use the NumPy port in `src/guabc/cec2017.py`, which reads the shift vectors, rotation matrices and
shuffle data from `external/cec2017/input_data`. `tools/port_check/` compares the port with the compiled official code
(copy `cec17_test_func.cpp` into that folder and run `build.cmd`, then `python tools/port_check/compare.py`).

CEC 2014 is evaluated through the official C code compiled as a DLL: `external/cec2014/cec14_dll.cpp` defines the
globals of the official code and exports one evaluation call; `build_dll.cmd` builds `cec14.dll` with the Microsoft
C++ build tools (adjust the path to `vcvars64.bat` if needed).
