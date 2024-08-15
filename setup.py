from pathlib import Path

from setuptools import setup, find_packages


with Path("README.md").open() as readme_file:
    readme = readme_file.read()

setup(
    name="wsiloader",
    version="0.0.2",
    author="Gaspar Faure",
    license="MIT",
    description="An efficient PyTorch dataloader for working with Whole-Slide Images",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples"]),
    url="https://github.com/gafaua/wsi-dataloader",
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
    ],
)
