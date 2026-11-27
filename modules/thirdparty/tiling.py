# Copied from https://github.com/BlenderNeko/ComfyUI_TiledKSampler (2fd9b05), less one assignment that nothing read. Used by tiled_sampler.py.
# Changed here: a tile is written back by the last two dimensions, and a mask of `padded` is one mask
# for every latent of the batch.
# And `mask_at_boundary` keeps the batch and channels of the mask it is given.

import torch
import itertools
import numpy as np

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def create_batches(n, iterable):
    groups = itertools.groupby(iterable, key= lambda x: (x[1], x[3]))
    for _, x in groups:
        for y in grouper(n, x):
            yield y


def get_slice(tensor, h, h_len, w, w_len):
    t = tensor.narrow(-2, h, h_len)
    t = t.narrow(-1, w, w_len)
    return t

def set_slice(tensor1,tensor2,  h, h_len, w, w_len, mask=None):
    if mask is not None:
        tensor1[...,h:h+h_len,w:w+w_len] = tensor1[...,h:h+h_len,w:w+w_len] * (1 - mask) +  tensor2 * mask
    else:
        tensor1[...,h:h+h_len,w:w+w_len] = tensor2

def get_tiles_and_masks_simple(steps, latent_shape, tile_height, tile_width):
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]
    tile_size_h = int(tile_height // 8)
    tile_size_w = int(tile_width // 8)

    h = np.arange(0,latent_size_h, tile_size_h)
    w = np.arange(0,latent_size_w, tile_size_w)

    def create_tile(hs, ws, i, j):
        h = int(hs[i])
        w = int(ws[j])
        h_len = min(tile_size_h, latent_size_h - h)
        w_len = min(tile_size_w, latent_size_w - w)
        return (h, h_len, w, w_len, steps, None)

    passes = [
        [[create_tile(h, w, i, j) for i in range(len(h)) for j in range(len(w))]],
    ]
    return passes

def get_tiles_and_masks_padded(steps, latent_shape, tile_height, tile_width):
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]

    tile_size_h = int(tile_height // 8)
    tile_size_h = int((tile_size_h // 4) * 4)
    tile_size_w = int(tile_width // 8)
    tile_size_w = int((tile_size_w // 4) * 4)

    mask_h = [0,tile_size_h // 4, tile_size_h - tile_size_h // 4, tile_size_h]
    mask_w = [0,tile_size_w // 4, tile_size_w - tile_size_w // 4, tile_size_w]
    masks = [[] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            mask = torch.zeros((1,1,tile_size_h, tile_size_w), dtype=torch.float32, device='cpu')
            mask[:,:,mask_h[i]:mask_h[i+1],mask_w[j]:mask_w[j+1]] = 1.0
            masks[i].append(mask)
    
    def create_mask(h_ind, w_ind, h_ind_max, w_ind_max, mask_h, mask_w, h_len, w_len):
        mask = masks[1][1]
        if not (h_ind == 0 or h_ind == h_ind_max or w_ind == 0 or w_ind == w_ind_max):
            return get_slice(mask, 0, h_len, 0, w_len)
        mask = mask.clone()
        if h_ind == 0 and mask_h:
            mask += masks[0][1]
        if h_ind == h_ind_max and mask_h:
            mask += masks[2][1]
        if w_ind == 0 and mask_w:
            mask += masks[1][0]
        if w_ind == w_ind_max and mask_w:
            mask += masks[1][2]
        if h_ind == 0 and w_ind == 0 and mask_h and mask_w:
            mask += masks[0][0]
        if h_ind == 0 and w_ind == w_ind_max and mask_h and mask_w:
            mask += masks[0][2]
        if h_ind == h_ind_max and w_ind == 0 and mask_h and mask_w:
            mask += masks[2][0]
        if h_ind == h_ind_max and w_ind == w_ind_max and mask_h and mask_w:
            mask += masks[2][2]
        return get_slice(mask, 0, h_len, 0, w_len)

    h = np.arange(0,latent_size_h, tile_size_h)
    h_shift = np.arange(tile_size_h // 2, latent_size_h - tile_size_h // 2, tile_size_h)
    w = np.arange(0,latent_size_w, tile_size_w)
    w_shift = np.arange(tile_size_w // 2, latent_size_w - tile_size_h // 2, tile_size_w)
    

    def create_tile(hs, ws, mask_h, mask_w, i, j):
        h = int(hs[i])
        w = int(ws[j])
        h_len = min(tile_size_h, latent_size_h - h)
        w_len = min(tile_size_w, latent_size_w - w)
        mask = create_mask(i,j,len(hs)-1, len(ws)-1, mask_h, mask_w, h_len, w_len)
        return (h, h_len, w, w_len, steps, mask)
    
    passes = [
        [[create_tile(h,       w,       True,  True,  i, j) for i in range(len(h))       for j in range(len(w))]],
        [[create_tile(h_shift, w,       False, True,  i, j) for i in range(len(h_shift)) for j in range(len(w))]],
        [[create_tile(h,       w_shift, True,  False, i, j) for i in range(len(h))       for j in range(len(w_shift))]],
        [[create_tile(h_shift, w_shift, False, False, i,j) for i in range(len(h_shift)) for j in range(len(w_shift))]],
    ]
    
    return passes

def mask_at_boundary(h, h_len, w, w_len, tile_size_h, tile_size_w, latent_size_h, latent_size_w, mask, device='cpu'):
    tile_size_h = int(tile_size_h // 8)
    tile_size_w = int(tile_size_w // 8)
    
    if (h_len == tile_size_h or h_len == latent_size_h) and (w_len == tile_size_w or w_len == latent_size_w):
        return h, h_len, w, w_len, mask
    h_offset = min(0, latent_size_h - (h + tile_size_h))
    w_offset = min(0, latent_size_w - (w + tile_size_w))
    new_mask = torch.zeros(((1,1) if mask is None else mask.shape[:-2]) + (tile_size_h, tile_size_w), dtype=torch.float32, device=device)
    new_mask[...,-h_offset:h_len if h_offset == 0 else tile_size_h, -w_offset:w_len if w_offset == 0 else tile_size_w] =  1.0 if mask is None else mask
    return h + h_offset, tile_size_h, w + w_offset, tile_size_w, new_mask

def get_tiles_and_masks_rgrid(steps, latent_shape, tile_height, tile_width, generator):

    def calc_coords(latent_size, tile_size, jitter):
        tile_coords = int((latent_size + jitter - 1) // tile_size + 1)
        tile_coords = [np.clip(tile_size * c - jitter, 0, latent_size) for c in range(tile_coords + 1)]
        tile_coords = [(c1, c2-c1) for c1, c2 in zip(tile_coords, tile_coords[1:])]
        return tile_coords
    
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]
    tile_size_h = int(tile_height // 8)
    tile_size_w = int(tile_width // 8)

    tiles_all = []

    for s in range(steps):
        rands = torch.rand((2,), dtype=torch.float32, generator=generator, device='cpu').numpy()

        jitter_w1 = int(rands[0] * tile_size_w)
        jitter_w2 = int(((rands[0] + .5) % 1.0) * tile_size_w)
        jitter_h1 = int(rands[1] * tile_size_h)
        jitter_h2 = int(((rands[1] + .5) % 1.0) * tile_size_h)

        tiles_h = [
            calc_coords(latent_size_h, tile_size_h, jitter_h1),
            calc_coords(latent_size_h, tile_size_h, jitter_h2)
        ]
        tiles_w = [
            calc_coords(latent_size_w, tile_size_w, jitter_w1),
            calc_coords(latent_size_w, tile_size_w, jitter_w2)
        ]

        tiles = []
        if s % 2 == 0:
            for i, h in enumerate(tiles_h[0]):
                for w in tiles_w[i%2]:
                    tiles.append((int(h[0]), int(h[1]), int(w[0]), int(w[1]), 1, None))
        else:
            for i, w in enumerate(tiles_w[0]):
                for h in tiles_h[i%2]:
                    tiles.append((int(h[0]), int(h[1]), int(w[0]), int(w[1]), 1, None))
        tiles_all.append(tiles)
    return [tiles_all]