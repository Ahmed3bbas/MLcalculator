import sys
from cx_Freeze import setup, Executable
import os

additional_mods = ['numpy.core._methods', 'numpy.lib.format','numpy._distributor_init',"multiprocessing","multiprocessing.pool"]
os.environ['TCL_LIBRARY'] = r"C:\Users\lenovo\AppData\Local\Programs\Python\Python35-32\tcl\tcl8.6"
os.environ['TK_LIBRARY'] = r"C:\Users\lenovo\AppData\Local\Programs\Python\Python35-32\tcl\tk8.6"

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["pygame","numpy","cv2","sklearn","skimage","os","math","scipy","multiprocessing","multiprocessing.pool","keras","collections","gzip"],
                     "include_files":["img",
                                      "pretrained",
                                      "data"],
                     'includes': additional_mods,
                     'excludes':["scipy.spatial.cKDTree"]}

# GUI applications require a different base on Windows (the default is for a
# console application
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "ML Calculator",
        version = "1.0",
        description = "simple machine learning calculator with basic operation [+,-,*]",
        options = {"build_exe": build_exe_options},
        executables = [Executable("Calculator.py", base=base)])
