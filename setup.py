import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON=ON",
        ]
        build_args = ["--config", "Release", "-j"]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

setup(
    name="piece_tokenizer",
    version="0.1.0",
    description="PieceTokenizer Python bindings",
    ext_modules=[CMakeExtension("piece_tokenizer")],
    cmdclass={"build_ext": CMakeBuild},
)
