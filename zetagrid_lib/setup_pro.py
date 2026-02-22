from setuptools import setup, Extension
from Cython.Build import cythonize

# 🛡️ COMMERCIAL BUILD SCRIPT
# This compiles the python source into a binary extension (.so/.pyd)
# making it unreadable to the user (IP Protection).

extensions = [
    Extension("zetagrid.engine", ["src/zetagrid/engine.pyx"]),
]

setup(
    name="zetagrid",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    package_dir={"": "src"},
)
