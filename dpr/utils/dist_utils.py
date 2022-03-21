#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for distributed model training
"""

import pickle
from typing import Tuple, List, Any

import torch
from torch import Tensor as T
import torch.distributed as dist


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(
    data: Any,
    group=None,
    max_size: int = None,
):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if max_size is None:
        # All processes must use the same buffer size
        max_size = enc_size + SIZE_STORAGE_BYTES
        max_sizes = gather(
            cfg=None,
            objects_to_sync=[max_size],
            buffer_size=1000,
        )[0]
        max_size = max(max_sizes)
        max_size = round(max_size * 1.1)  # take 110%

    elif enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            f'encoded data exceeds max_size, this can be fixed by increasing '
            f'buffer size: {enc_size}'
        )

    rank = get_rank()
    world_size = get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def gather(
    cfg,
    objects_to_sync: List[Any],
    buffer_size: int = None,
) -> List[Tuple]:
    """
    Helper function to gather arbitrary objects.

    Args:
        objects_to_sync (List[Any]): List of any arbitrary objects. This list
            should be of the same size across all processes.

    Returns
        gathered_objects (List[Tuple]):  List of size `num_objects`, where
            `num_objects` is the number of objects in `objects_to_sync` input.
            Each element in this list is a tuple of gathered objects from
            multiple processes (of the same object).
    """
    if not dist.is_initialized():
        return [[obj] for obj in objects_to_sync]

    local_rank = dist.get_rank()
    distributed_world_size = dist.get_world_size()

    if distributed_world_size > 1:
        # For tensors that reside on GPU, we first need to detach it from its
        # computation graph, clone it, and then transfer it to CPU
        on_gpus = []
        copied_objects_to_sync = []
        for object in objects_to_sync:
            if isinstance(object, T) and object.is_cuda:
                on_gpus.append(1)
                copied_object_to_sync = torch.empty_like(  # clone, detach and transfer to CPU
                    object, device="cpu"
                ).copy_(object).detach_()
                copied_objects_to_sync.append(copied_object_to_sync)
            else:
                on_gpus.append(0)
                copied_objects_to_sync.append(object)

        global_objects_to_sync = all_gather_list(
            [local_rank, copied_objects_to_sync],
            max_size=buffer_size,
        )
        # Sort gathered objects according to ranks, so that all processes
        # will receive the same objects in the same order
        global_objects_to_sync = sorted(global_objects_to_sync, key=lambda x: x[0])

        gathered_objects = []
        for rank, items in global_objects_to_sync:
            if rank == local_rank:
                gathered_objects.append(objects_to_sync)  # not the copied ones
            else:
                # `items` is a list of objects from `local_rank=rank`
                # If any object originally resides on GPU, we need
                # to transfer it back
                assert len(items) == len(on_gpus)
                copied_items = []
                for item, on_gpu in zip(items, on_gpus):
                    if on_gpu:
                        item = item.to(cfg.device)
                    copied_items.append(item)
                gathered_objects.append(copied_items)

    else:
        gathered_objects = [objects_to_sync]

    gathered_objects = list(zip(*gathered_objects))
    return gathered_objects
