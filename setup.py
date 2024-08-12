from setuptools import setup

setup(
    name="wsiloader",
    version="0.0.1",
    author="Gaspar Faure",
    description="A PyTorch efficient dataloader for working with Whole-Slide Images",
    py_modules=["wsiloader"],
    url="https://github.com/gafaua/wsi-dataloader",
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
    ],
)
