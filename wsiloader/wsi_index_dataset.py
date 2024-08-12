from pathlib import Path
from typing import Callable, Iterator, List

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


class WSIIndexDataset(Dataset):
    def __init__(self,
                 wsi_paths: List[str | Path],
                 patch_generator: Callable[[str | Path], Iterator[Image.Image | Tensor | np.ndarray]],
                 transforms = None,
                 ):
        """
        Dataset class for indexing a list of WSIs. This dataset, coupled with a regular PyTorch Dataloader
        enables fast and random patch access from all provided WSIs.
        Transforms (applied to patches on CPU) can be passed with the `transforms` parameter.

        Args:
            wsi_paths (List[str  |  Path]): List of WSI paths to build the dataset from.
            patch_generator (Callable[[str | Path], Iterator[Image.Image  |  Tensor  |  np.ndarray]]):
                                            Generator function, taking a path to a WSI as input and returning
                                            an Iterator over all desired patches from the WSI.
            transforms (_type_, optional): CPU only transforms to apply to patches. Defaults to None.
        """
        self.wsi_paths = wsi_paths
        self.patch_generator = patch_generator
        self.transforms = transforms

        self.index_slides()

    def index_slides(self):
        self.patch_extractors: List[Iterator[Image.Image | Tensor]] = []
        self.total_cnt: int = 0

        # Maps for fast patch retrieval from global indices
        self.idx_to_wsi_id: List[int] = []
        self.wsi_id_to_first_idx: List[int] = []

        wsi_id = 0
        pbar = tqdm(self.wsi_paths, "Building WSI index", ncols=120)

        for path in pbar:
            patches = self.patch_generator(path)

            self.wsi_id_to_first_idx.append(self.total_cnt)
            self.total_cnt += len(patches)
            self.patch_extractors.append(patches)
            self.idx_to_wsi_id += [wsi_id] * len(patches)
            wsi_id += 1

            pbar.set_postfix_str(f"[{len(patches)} > {self.total_cnt:,} indexed patches]", refresh=True)

    def __get_patch_at_idx(self, idx):
        wsi_id = self.idx_to_wsi_id[idx]
        patch_idx = idx - self.wsi_id_to_first_idx[wsi_id]

        return self.patch_extractors[wsi_id][patch_idx]

    def __getitem__(self, index):
        patch = self.__get_patch_at_idx(index)

        # Assuming that transforms expect a Tensor
        if isinstance(patch, np.ndarray):
            patch = to_tensor(patch.copy())
        elif isinstance(patch, Image.Image):
            patch = to_tensor(patch)

        if self.transforms is None:
            # Convert to tensor for GPU processing
            return patch
        else:
            return self.transforms(patch)

    def __len__(self):
        return self.total_cnt
