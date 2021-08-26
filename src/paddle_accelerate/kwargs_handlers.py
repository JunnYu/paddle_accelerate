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

import copy
from dataclasses import dataclass


class KwargsHandler:
    """
    Internal mixin that implements a :obj:`to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    strategy: str = None
    last_comm_buffer_size: int = 1
    comm_buffer_size: int = 25
    find_unused_parameters: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    init_loss_scaling: float = 65536.0
    incr_ratio: float = 2.0
    decr_ratio: float = 0.5
    incr_every_n_steps: int = 2000
    decr_every_n_nan_or_inf: int = 2
    use_dynamic_loss_scaling: bool = True
    enabled: bool = True
