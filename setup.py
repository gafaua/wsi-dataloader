from setuptools import setup, find_packages

setup(
    name="wsiloader",
    version="0.0.1",
    author="Gaspar Faure",
    license="MIT",
    description="An efficient PyTorch dataloader for working with Whole-Slide Images",
    packages=find_packages(exclude=["examples"]),
    url="https://github.com/gafaua/wsi-dataloader",
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
    ],
)
