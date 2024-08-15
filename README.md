## WSI Dataloader

The WSI Dataloader library offers a simple implementation that enables online access to whole-slide images (WSI) during the training of deep learning models. In most machine learning frameworks designed for WSI analysis the very large WSI files are split into patches, usually for memory limitation reasons. Generating patch datasets can be long, resource-consuming and sometimes impossible when working with limited storage constraints.

The `WSIDataloader` class offers an alternative solution to generating patch datasets. It is a [PyTorch](https://pytorch.org/) based implementation encapsulating a `Dataloader` and a `Dataset`. It enables online patch extraction across a given list of WSI files, directly during training. It supports all usual Dataloader parameters to parallelize and speed up data loading (`num_workers`, `prefetch_ratio`). 

### Supported features
- Random patch sampling over a list of WSIs 
- Support for data loading over multiple workers
- CUDA acceleration for data augmentation (more on this below)
- User-defined patch definition for flexibility (the user how patches should be extracted from WSIs)
- Support for standard PyTorch Dataloader arguments
- Easy to adapt to your own pipeline. The `wsiloader` library only consists of 2 classes: [`WSIDataloader`](./wsiloader/wsi_dataloader.py) and [`WSIIndexDataset`](./wsiloader/wsi_index_dataset.py), making it easy to create custom classes inheriting from these base classes.

#### CUDA acceleration for data augmentation
The `WSIDataloader` class supports CUDA acceleration for transforms application (data augmentation). When the `transforms_device` parameter is set to "cpu", the default Dataloader behaviour is used and the transforms are applied in the Dataloader workers. When it is set to "cuda", the patches are first loaded using the Dataloader workers, and then transforms are sequentially applied on GPU. This decoupling is necessary due to CUDA's inability to be used in multiprocessing contexts. Depending on the nature of the required transforms, using CUDA for data augmentation can substantially reduce a training loop's iteration time. The `basic_example.ipynb` notebook provides an example.

### Installation

Install the `wsiloader` library using pip from PyPI:
```sh
$ pip install wsiloader
```
or from GitHub
```sh
$ pip install git+https://github.com/gafaua/wsi-dataloader/tree/main
```
Confirm the installation by importing the `WSIDataloader` class:
```sh
$ python -c "from wsiloader import WSIDataloader"
```

### Examples
Example notebooks can be found in the [examples](./examples/) directory. We recommend to take a look at these to get a better idea of how to take advantage of the `wsiloader` library for your pipeline.
