from pathlib import Path
from typing import Callable, Iterator, List

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.sampler import Sampler

from wsiloader.wsi_index_dataset import WSIIndexDataset


class WSIDataloader(Sampler):
    def __init__(self,
                 wsi_paths: List[str | Path],
                 patch_generator: Callable[[str | Path], Iterator[Image.Image | Tensor | np.ndarray]],
                 transforms=None,
                 transforms_device="cpu",
                 collate_function=None,
                 **data_loader_kwargs,
                 ):
        """
        Wrapper class for PyTorch's Dataloader and Dataset classes to decouple data loading and transforms execution when applied
        to patches extracted from Whole-Slide Images. This enables both parallelization of data loading using multiple
        Dataloader workers and cuda acceleration for transforms application. When the `transforms_device` parameter is set to "cuda",
        the transforms are sequentially applied on GPU, when set to "cpu", the default Dataloader behaviour is used and the
        transforms are applied in the Dataloader workers.
        See examples of how to use at: 

        Args:
            wsi_paths (List[str | Path]): A list of paths to all WSIs to include in the dataset.
            patch_generator (Callable[[str | Path], Iterator[Image.Image  |  Tensor  |  np.ndarray]]): Generator function taking
                                           the path to a WSI as input and returning an iterator over patches extracted from the WSI.
            transforms (optional): Transforms to apply to each element of a batch. Defaults to None.
            transforms_device (str, optional): Device used for executing tranforms to the batch. When set to "cpu", the transforms
                                               are executed by the DataLoader's workers. When set to "cuda", the transforms are executed
                                               after the patches have been collected by the DataLoader's workers. Defaults to "cpu".
            collate_function (optional): Collate function used to collate the elements of the batch after tranforms are
                                         applied. When set to `None`, defaults to torch's `default_collate` function. Defaults to None.
            **data_loader_kwargs: PyTorch Dataloader keyword arguments
        """
        self.transforms_device = transforms_device

        if "cuda" in self.transforms_device:
            self.transforms = transforms
            self.index = WSIIndexDataset(wsi_paths, patch_generator=patch_generator, transforms=None)
        else:
            # Transforms are executed on CPU, they can be exected by the DataLoader's workers
            self.transforms = None
            self.index = WSIIndexDataset(wsi_paths, patch_generator=patch_generator, transforms=transforms)

        self.loader = DataLoader(self.index, **data_loader_kwargs)

        if collate_function is None:
            self.collate = default_collate
        else:
            self.collate = collate_function

    def reset_index(self):
        self.index.index_slides()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if self.transforms is None:
                yield batch
            else:
                batch = batch.to(self.transforms_device)
                batch = [self.transforms(elem) for elem in batch]
                batch = self.collate(batch)

                yield batch
