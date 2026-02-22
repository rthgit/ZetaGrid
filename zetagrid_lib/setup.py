from setuptools import setup, find_packages

setup(
    name="zetagrid",
    version="0.1.0",
    description="The Unleashed PyTorch Accelerator for Single-Node Training",
    author="Antigravity",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "transformers",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
