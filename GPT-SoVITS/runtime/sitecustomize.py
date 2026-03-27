import os
import sys


RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNTIME_DIR)
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
GPT_SOVITS_DIR = os.path.join(PROJECT_ROOT, "GPT_SoVITS")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if GPT_SOVITS_DIR not in sys.path:
    sys.path.insert(0, GPT_SOVITS_DIR)
