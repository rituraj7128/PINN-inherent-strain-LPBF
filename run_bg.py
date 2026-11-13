"""Runs a training script without a console window (used by the START_*_BACKGROUND.bat files with pythonw).
Usage: pythonw run_bg.py <script.py> <logfile>. All output and any error go to <logfile>."""
import os, sys, runpy, traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))
f = open(sys.argv[2], "a", buffering=1); sys.stdout = sys.stderr = f
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except BaseException:
    traceback.print_exc()
