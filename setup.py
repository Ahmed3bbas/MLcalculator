import sys
from cx_Freeze import setup, Executable
import os

additional_mods = ['numpy.core._methods', 'numpy.lib.format','numpy._distributor_init',"multiprocessing","multiprocessing.pool"]
os.environ['TCL_LIBRARY'] = r"C:\Users\lenovo\AppData\Local\Programs\Python\Python35-32\tcl\tcl8.6"
os.environ['TK_LIBRARY'] = r"C:\Users\lenovo\AppData\Local\Programs\Python\Python35-32\tcl\tk8.6"

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["pygame","numpy","cv2","sklearn","os","math","scipy","multiprocessing","multiprocessing.pool"],
                     "include_files":["calculator_svm.pkl",
                                        "prdicted_image.jpg",
                                      "trainClassfier.py",
                                      "data.py",
                                      "operator_features.npz",
                                      "operator_Labels.npy"],
                     'includes': additional_mods,
                     'excludes':["scipy.spatial.cKDTree"]}

# GUI applications require a different base on Windows (the default is for a
# console application
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "ML Calculator",
        version = "0.1",
        description = "My GUI Calculator With ML!",
        options = {"build_exe": build_exe_options},
        executables = [Executable("Calculator.pyw", base=base)])
