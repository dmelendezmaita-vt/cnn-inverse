from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="deep-learning-toolkit",
    version="0.1.0",
    description="DL-Kit: Deep Learning tool-Kit",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Johann Rudi, Harrison J Goldwyn",
    license="GPL-3.0-only",
    python_requires=">=3.8",
    packages=find_packages(exclude=["_*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.26,<2",
        "prettytable>=3",
        "torch>=2",
        "tqdm>=4",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
