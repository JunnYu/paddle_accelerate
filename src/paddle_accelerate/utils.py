# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import random
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import paddle

from .state import AcceleratorState, DistributedType


class RNGType(Enum):
    PADDLE = "paddle"
    CUDA = "cuda"
    GENERATOR = "generator"


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def get_cpu_rng_state():
    return paddle.fluid.core.default_cpu_generator().get_state()


def set_cpu_rng_state(rng_state):
    return paddle.fluid.core.default_cpu_generator().set_state(rng_state)


def get_cuda_rng_state():
    return paddle.fluid.core.default_cuda_generator(0).get_state()


def set_cuda_rng_state(state):
    return paddle.fluid.core.default_cuda_generator(0).set_state(state)


def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator=None):
    # Get the proper rng state
    if rng_type == RNGType.PADDLE:
        rng_state = get_cpu_rng_state()
    elif rng_type == RNGType.CUDA:
        rng_state = get_cuda_rng_state()
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, "Need a generator to synchronize its seed."
        rng_state = generator.get_state()

    # Broadcast the rng state from device 0 to other devices
    state = AcceleratorState()
    if state.distributed_type == DistributedType.MULTI_GPU:
        paddle.distributed.broadcast(rng_state, 0)

    # Set the broadcast rng state
    if rng_type == RNGType.PADDLE:
        set_cpu_rng_state(rng_state)
    elif rng_type == RNGType.CUDA:
        set_cuda_rng_state(rng_state)
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)


def synchronize_rng_states(
    rng_types: List[Union[str, RNGType]],
    generator: Optional[paddle.fluid.core_avx.Generator] = None,
):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    # There is no direct check whether an object if of type namedtuple sadly, this is a workaround.
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # Can instantiate a namedtuple from a generator directly, contrary to a tuple/list.
        return type(obj)(*list(generator))
    return type(obj)(generator)


def extract_model_from_parallel(model):

    options = (paddle.DataParallel,)

    while isinstance(model, options):
        model = model._layers
    return model


def _gpu_gather(tensor):
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (_gpu_gather(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: _gpu_gather(v) for k, v in tensor.items()})
    elif not isinstance(tensor, paddle.Tensor):
        raise TypeError(
            f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )
    output_tensors = [
        tensor.clone() for _ in range(paddle.distributed.get_world_size())
    ]
    paddle.distributed.all_gather(output_tensors, tensor)
    return paddle.concat(output_tensors, axis=0)


def gather(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from all devices.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to gather.

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        return _gpu_gather(tensor)
    else:
        return tensor


def pad_across_processes(tensor, axis=0, pad_index=0, pad_first=False):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so they
    can safely be gathered.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to gather.
        dim (:obj:`int`, `optional`, defaults to 0):
            The dimension on which to pad.
        pad_index (:obj:`int`, `optional`, defaults to 0):
            The value with which to pad.
        pad_first (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to pad at the beginning or the end.
    """
    if isinstance(tensor, (list, tuple)):
        return honor_type(
            tensor,
            (pad_across_processes(t, axis=axis, pad_index=pad_index) for t in tensor),
        )
    elif isinstance(tensor, dict):
        return type(tensor)(
            {
                k: pad_across_processes(v, axis=axis, pad_index=pad_index)
                for k, v in tensor.items()
            }
        )
    elif not isinstance(tensor, paddle.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if axis >= tensor.ndim:
        return tensor

    # Gather all sizes
    size = paddle.to_tensor(tensor.shape, place=tensor.place).unsqueeze(0)
    sizes = gather(size).cpu()
    # Then pad to the maximum size
    max_size = max(s[axis] for s in sizes)
    if max_size == tensor.shape[axis]:
        return tensor

    old_size = tensor.shape
    new_size = list(old_size)
    new_size[axis] = max_size
    new_tensor = (
        paddle.zeros(tuple(new_size), dtype=tensor.dtype, place=tensor.place)
        + pad_index
    )
    if pad_first:
        indices = tuple(
            slice(max_size - old_size[axis], max_size) if i == axis else slice(None)
            for i in range(len(new_size))
        )
    else:
        indices = tuple(
            slice(0, old_size[axis]) if i == axis else slice(None)
            for i in range(len(new_size))
        )
    new_tensor[indices] = tensor
    return new_tensor


def wait_for_everyone():
    """
    Introduces a blocking point in the script, making sure all processes have reached this point before continuing.

    Warning::

        Make sure all processes will reach this instruction otherwise one of your processes will hang forever.
    """
    if AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        paddle.distributed.barrier()


def save(obj, f):
    """
    Save the data to disk. Use in place of :obj:`torch.save()`.

    Args:
        obj: The data to save
        f: The file (or file-like object) to use to save the data
    """
    if AcceleratorState().local_process_index == 0:
        paddle.save(obj, f)
