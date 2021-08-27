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

import gc
import logging
from contextlib import contextmanager
from typing import List, Optional

import paddle

from .kwargs_handlers import (
    DistributedDataParallelKwargs,
    GradScalerKwargs,
    KwargsHandler,
)
from .optimizer import AcceleratedOptimizer
from .state import AcceleratorState, DistributedType
from .utils import (
    extract_model_from_parallel,
    gather,
    pad_across_processes,
    save,
    wait_for_everyone,
)

logger = logging.getLogger(__name__)


class Accelerator:
    def __init__(
        self,
        fp16: bool = None,
        kwargs_handlers: Optional[List[KwargsHandler]] = None,
    ):
        self.state = AcceleratorState(fp16=fp16, _from_accelerator=True)
        # Kwargs handlers
        self.scaler_handler = None
        self.ddp_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(
                    handler, KwargsHandler
                ), f"Unsupported kwargs handler passed: {handler}."
                if isinstance(handler, DistributedDataParallelKwargs):
                    if self.ddp_handler is not None:
                        raise ValueError(
                            "You can only pass one `DistributedDataParallelKwargs` in `kwargs_handler`."
                        )
                    else:
                        self.ddp_handler = handler
                elif isinstance(handler, GradScalerKwargs):
                    if self.scaler_handler is not None:
                        raise ValueError(
                            "You can only pass one `GradScalerKwargs` in `kwargs_handler`."
                        )
                    else:
                        self.scaler_handler = handler

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if self.state.use_fp16:
            self.native_amp = True
            kwargs = (
                self.scaler_handler.to_kwargs()
                if self.scaler_handler is not None
                else {}
            )
            self.scaler = paddle.amp.GradScaler(**kwargs)

        # Internal references to the training objects
        self._optimizers = []
        self._models = []


    @property
    def distributed_type(self):
        return self.state.distributed_type

    @property
    def num_processes(self):
        return self.state.num_processes

    @property
    def local_process_index(self):
        return paddle.distributed.get_rank()

    @property
    def device(self):
        return self.state.device

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.local_process_index == 0

    @property
    def use_fp16(self):
        return self.state.use_fp16

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.
        """
        yield from self._goes_first(self.is_local_main_process)

    def _goes_first(self, is_main):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    def print(self, *args, **kwargs):
        """
        Use in replacement of :obj:`print()` to only print once per server.
        """
        if self.is_local_main_process:
            print(*args, **kwargs)

    def _prepare_one(self, obj):
        if isinstance(obj, paddle.nn.Layer):
            self._models.append(obj)
            return self.prepare_model(obj)
        elif isinstance(obj, paddle.optimizer.Optimizer):
            if len(self._models) == 1:
                obj._parameter_list = list(self._models[0].parameters())
            optimizer = self.prepare_optimizer(obj)
            self._optimizers.append(optimizer)
            return optimizer
        else:
            return obj

    def prepare(self, *args):
        result = tuple(self._prepare_one(obj) for obj in args)
        return result if len(result) > 1 else result[0]

    def prepare_model(self, model):
        if self.distributed_type == DistributedType.MULTI_GPU:
            kwargs = (
                self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
            )
            model = paddle.DataParallel(model, **kwargs)

        if self.native_amp:
            model.__call__ = paddle.amp.auto_cast(
                enable=True, custom_white_list=["layer_norm", "softmax", "gelu"]
            )(model.__call__)

        return model

    def prepare_optimizer(self, optimizer):
        return AcceleratedOptimizer(optimizer, scaler=self.scaler)

    def backward(self, loss, **kwargs):
        """
        Use :obj:`accelerator.backward(loss)` in lieu of :obj:`loss.backward()`.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def gather(self, tensor):
        """
        Gather the values in `tensor` accross all processes and concatenate them on the first dimension. Useful to
        regroup the predictions from all processes when doing evaluation.

        Note:
            This gather happens in all processes.

        Args:
            tensor (:obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`):
                The tensors to gather across all processes.

        Returns:
            :obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`: The gathered tensor(s). Note
            that the first dimension of the result is `num_processes` multiplied by the first dimension of the input
            tensors.
        """
        return gather(tensor)

    def pad_across_processes(self, tensor, axis=0, pad_index=0, pad_first=False):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.

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
        return pad_across_processes(
            tensor, axis=axis, pad_index=pad_index, pad_first=pad_first
        )

    def unwrap_model(self, model):
        """
        Unwraps the :obj:`model` from the additional layer possible added by :meth:`~accelerate.Accelerator.prepare`.
        Useful before saving the model.

        Args:
            model (:obj:`torch.nn.Module`):
                The model to unwrap.
        """
        return extract_model_from_parallel(model)

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.
        """
        wait_for_everyone()

    def save(self, obj, f):
        """
        Save the object passed to disk once per machine. Use in place of :obj:`torch.save`.

        Args:
            obj: The object to save.
            f (:obj:`str` or :obj:`os.PathLike`):
                Where to save the content of :obj:`obj`.
        """
        save(obj, f)

    def free_memory(self):
        """
        Will release all references to the internal objects stored and call the garbage collector. You should call this
        method between two trainings with different models/optimizers.
        """
        self._optimizers = []
        self._models = []
        gc.collect()

    def _get_named_parameters(self, *args):
        named_parameters = {}
        for obj in args:
            if isinstance(obj, paddle.nn.Layer):
                obj = extract_model_from_parallel(obj)
                named_parameters.update({n: p for n, p in obj.named_parameters()})
        return named_parameters

    def _get_devices(self, *args):
        model_device = None
        optimizer_device = None
        for obj in args:
            # Loop through model parameters and stop at the first once we have its device.
            if isinstance(obj, paddle.nn.Layer):
                for param in obj.parameters():
                    model_device = param.place
                    break
            # Loop through optimizer parameters groups and stop at the first once we have its device.
            if isinstance(obj, paddle.optimizer.Optimizer):
                optimizer_device = obj._parameter_list[0].place
        return (model_device, optimizer_device)

    def get_state_dict(self, model):
        model = self.unwrap_model(model)
        state_dict = model.state_dict()

        for k in state_dict:
            if state_dict[k].dtype == paddle.float16:
                state_dict[k] = state_dict[k].astype(paddle.float32)
        return state_dict

    @contextmanager
    def autocast(self):
        """
        Will apply automatic mixed-precision inside the block inside this context manager, if it is enabled. Nothing
        different will happen otherwise.
        """
        if self.native_amp:
            autocast_context = paddle.amp.auto_cast(
                enable=True, custom_white_list=["layer_norm", "softmax", "gelu"]
            )
            autocast_context.__enter__()
            yield
            autocast_context.__exit__()
        else:
            yield
