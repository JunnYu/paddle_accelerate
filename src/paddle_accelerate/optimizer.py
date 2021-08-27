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

import paddle

from .state import AcceleratorState


class AcceleratedOptimizer(paddle.optimizer.Optimizer):
    def __init__(self, optimizer, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.state = AcceleratorState()

    @property
    def _parameter_list(self):
        return self.optimizer._parameter_list

    @_parameter_list.setter
    def _parameter_list(self, parameter_list):
        self.optimizer._parameter_list = parameter_list

    def load_state_dict(self, state_dict):
        self.optimizer.set_state_dict(state_dict)

    def set_state_dict(self, state_dict):
        self.optimizer.set_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self):
        self.optimizer.clear_grad()

    def clear_grad(self):
        self.optimizer.clear_grad()

    def step(self, scaled=None):
        if self.scaler is not None and scaled is not None:
            self.scaler.minimize(self.optimizer, scaled)
        else:
            self.optimizer.step()
