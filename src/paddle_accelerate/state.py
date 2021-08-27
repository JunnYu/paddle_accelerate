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
import os
from distutils.util import strtobool
from enum import Enum

import paddle.distributed as dist


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def parse_flag_from_env(key, default=False):
    value = os.environ.get(key, str(default))
    return (
        strtobool(value) == 1
    )  # As its name indicates `strtobool` actually returns an int...


class DistributedType(str, Enum):
    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    SINGLE_GPU = "SINGLE_GPU"
    MULTI_GPU = "MULTI_GPU"


class AcceleratorState:
    _shared_state = {}

    def __init__(self, fp16: bool = None, _from_accelerator: bool = False):
        self.__dict__ = self._shared_state
        if not getattr(self, "initialized", False):
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )
            elif int(os.environ.get("LOCAL_RANK", -1)) != -1:
                self.distributed_type = DistributedType.MULTI_GPU
                dist.init_parallel_env()
                self.num_processes = dist.get_rank() + 1
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                self.device = f"gpu:{self.local_process_index}"
                self.use_fp16 = (
                    parse_flag_from_env("USE_FP16", False) if fp16 is None else fp16
                )
            else:
                self.distributed_type = DistributedType.SINGLE_GPU
                self.num_processes = dist.get_rank() + 1
                self.local_process_index = 0
                self.device = f"gpu:{self.local_process_index}"
                self.use_fp16 = (
                    parse_flag_from_env("USE_FP16", False) if fp16 is None else fp16
                )

            self.initialized = True

    def __repr__(self):
        repr = (
            f"Distributed environment: {self.distributed_type}\n"
            f"Num processes: {self.num_processes}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
            f"Use FP16 precision: {self.use_fp16}\n"
        )
        return repr
