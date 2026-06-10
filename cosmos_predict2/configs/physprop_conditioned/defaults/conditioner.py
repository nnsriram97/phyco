# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.vid2vid.defaults.conditioner import Vid2VidCondition, Vid2VidConditioner
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
import copy
from typing import Any, Tuple

@dataclass(frozen=True)
class PhysPropConditionedCondition(Vid2VidCondition):
    physprop: Optional[torch.Tensor] = None
    pred_physprop: Optional[torch.Tensor] = None
    first_depth_frame: Optional[torch.Tensor] = None


class PhysPropConditionedConditioner(Vid2VidConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> PhysPropConditionedCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "physprop" in batch, "PhysPropConditionalConditioner requires 'physprop' in batch"
        output["physprop"] = batch["physprop"]
        if "first_depth_frame" in batch:
            output["first_depth_frame"] = batch["first_depth_frame"]
        return PhysPropConditionedCondition(**output)

    # def get_condition_uncondition(
    #     self,
    #     data_batch: Dict,
    # ) -> Tuple[Any, Any]:
    #     """
    #     Custom condition/uncondition for physprop using 1-physprop instead of zeroing
    #     """
    #     cond_dropout_rates, dropout_rates = {}, {}
    #     for emb_name, embedder in self.embedders.items():
    #         cond_dropout_rates[emb_name] = 0.0
    #         if emb_name == "physprop":
    #             dropout_rates[emb_name] = 0.0
    #         else:
    #             dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

    #     # Create modified batch for un_condition with 1-physprop
    #     data_batch_uncond = copy.deepcopy(data_batch)
    #     if "physprop" in data_batch_uncond and self.embedders["physprop"].dropout_rate > 1e-4:
    #         data_batch_uncond["physprop"] = 1.0 - data_batch_uncond["physprop"]

    #     condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
    #     un_condition: Any = self(data_batch_uncond, override_dropout_rate=dropout_rates)
    #     return condition, un_condition
