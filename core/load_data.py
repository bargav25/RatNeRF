import cv2
import os, sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect
import h5py, math
import random
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate

from .dataset import *

class RandIntGenerator:
    '''
    RandomInt generator that ensures all n data will be
    sampled at least one in every n iteration.
    '''

    def __init__(self, n, generator=None):
        self._n = n
        self.generator = generator

    def __iter__(self):

        if self.generator is None:
            # TODO: this line is buggy for 1.7.0 ... but has to use this for 1.9?
            #       it induces large memory consumptions somehow
            generator = torch.Generator(device=torch.tensor(0.).device)
            #generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        yield from torch.randperm(self._n, generator=generator)

    def __len__(self):
        return self._n


class RayImageSampler(Sampler):
    '''
    TODO: does this work with ConcatDataset?
    TODO: handle train/val
    '''

    def __init__(self, data_source, N_images=1024,
                 N_iter=None, generator=None):
        self.data_source = data_source
        self.N_images = N_images
        self._N_iter = N_iter
        self.generator = generator

        if self._N_iter is None:
            self._N_iter = len(self.data_source)

        self.sampler = RandIntGenerator(n=len(self.data_source))

    def __iter__(self):

        sampler_iter = iter(self.sampler)
        batch = []
        for i in range(self._N_iter):
            # get idx until we have N_images in batch
            while len(batch) < self.N_images:
                try:
                    idx = next(sampler_iter)
                except StopIteration:
                    sampler_iter = iter(self.sampler)
                    idx = next(sampler_iter)
                batch.append(idx.item())

            # return and clear batch cache
            yield np.sort(batch)
            batch = []

    def __len__(self):
        return self._N_iter

def ray_collate_fn(batch):

    batch = default_collate(batch)
    # default collate results in shape (N_images, N_rays_per_images, ...)
    # flatten the first two dimensions.
    batch = {k: batch[k].flatten(end_dim=1) for k in batch}
    batch['rays'] = torch.stack([batch['rays_o'], batch['rays_d']], dim=0)
    return batch



def load_data(args):

    dataset = BaseH5Dataset(h5_path="data/rats/rat7mdata.h5")
    
    # Main loop controls the iteration, so simply set N_iter to something > args.n_iters
    sampler = RayImageSampler(dataset, N_images=args.N_sample_images,
                              N_iter=args.n_iters + 10)
    # initialize dataloader
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=ray_collate_fn)
                            
    data_attrs = dataset.get_meta()
    render_data = dataset.get_render_data()

    return dataloader, render_data, data_attrs

