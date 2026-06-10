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

from hydra.core.config_store import ConfigStore

from cosmos_predict2.callbacks.device_monitor import DeviceMonitor
from cosmos_predict2.callbacks.grad_clip import GradClip
from cosmos_predict2.callbacks.iter_speed import IterSpeed
from cosmos_predict2.callbacks.loss_log import LossLog
from cosmos_predict2.callbacks.valvid_saver import ValidationVideoSaver
from imaginaire.callbacks.manual_gc import ManualGarbageCollection
from imaginaire.lazy_config import PLACEHOLDER
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils.callback import LowPrecisionCallback

BASIC_CALLBACKS = dict(
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n="${trainer.logging_iter}",
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    loss_log=L(LossLog)(),
    grad_clip=L(GradClip)(clip_norm=1.0),
)


BASE_VID_SAVE_CALLBACKS = dict(
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n="${trainer.logging_iter}",
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    loss_log=L(LossLog)(),
    grad_clip=L(GradClip)(clip_norm=1.0),
    valvid_saver=L(ValidationVideoSaver)(),
)

PHYSPROP_CALLBACKS = dict(
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
    ),
    device_monitor=L(DeviceMonitor)(
        every_n="${trainer.logging_iter}",
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    loss_log=L(LossLog)(),
    grad_clip=L(GradClip)(clip_norm=1.0),
    valvid_saver=L(ValidationVideoSaver)(),
)

PHYSPROP_MEMORY_SAFE_CALLBACKS = dict(
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
        hit_thres=10,  # More aggressive CUDA sync for memory safety
    ),
    device_monitor=L(DeviceMonitor)(
        every_n=50,  # More frequent memory monitoring
        log_memory_detail=True,
    ),
    manual_gc=L(ManualGarbageCollection)(
        every_n=1,  # Aggressive garbage collection after every iteration
        warm_up=2,  # Shorter warm-up to start GC earlier
    ),
    loss_log=L(LossLog)(),
    grad_clip=L(GradClip)(
        clip_norm=0.5,  # Lower gradient clipping for stability
        force_finite=True,
    ),
    valvid_saver=L(ValidationVideoSaver)(),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="basic_vid_save", node=BASE_VID_SAVE_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="physprop", node=PHYSPROP_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="physprop_memory_safe", node=PHYSPROP_MEMORY_SAFE_CALLBACKS)
