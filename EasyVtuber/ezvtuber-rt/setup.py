"""Setup script for ezvtuber-rt library.

A real-time virtual tuber animation library powered by TensorRT and ONNX.
Provides ONNX models and integrated code for THA3, RIFE, Waifu2x, and Real-ESRGAN.

Note: C++ extensions (rgba_utils, ffmpeg_codec) are kept in the codebase but not
built by default. See ezvtb_rt/cpp/ for source files if manual building is needed.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies required for all installations
core_dependencies = [
    "numpy",
    "opencv-python",
    "tqdm",
    "onnx",
    "onnxruntime-directml",
    "pycuda",
    "brotli",  # For cache compression
    "pyanime4k==3.1.0" # For Anime4K super-resolution
]

setup(
    name="ezvtuber-rt",
    version="0.1.0",
    author="zpeng11",
    author_email="",
    description="Real-time virtual tuber animation library with TensorRT and ONNX support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zpeng11/ezvtuber-rt",
    project_urls={
        "Bug Tracker": "https://github.com/zpeng11/ezvtuber-rt/issues",
        "Source Code": "https://github.com/zpeng11/ezvtuber-rt",
    },
    packages=find_packages(exclude=["test", "test.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=core_dependencies,
    include_package_data=True,
    package_data={
        "ezvtb_rt": ["*.py"],
    },
    zip_safe=False,
    keywords=[
        "vtuber",
        "real-time",
        "animation",
        "tensorrt",
        "onnx",
        "deeplearning",
        "computer-vision",
        "gpu-acceleration",
        "frame-interpolation",
        "super-resolution",
    ],
)
