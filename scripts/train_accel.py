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

import argparse
import importlib
import os
import signal
import functools

from loguru import logger as logging
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch

from imaginaire.config import Config, pretty_print_overrides
from imaginaire.lazy_config import instantiate, LazyConfig
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils.config_helper import get_config_module, override
from imaginaire.utils import misc, log, callback
from imaginaire.utils.checkpointer import Checkpointer
from imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from imaginaire.utils import ema
from imaginaire.model import ImaginaireModel


class AccelerateTrainer:
    """Accelerate-based trainer for Cosmos Predict2 models with DeepSpeed ZeRO-3 support."""
    
    def __init__(self, config: Config):
        """Initialize the Accelerate trainer.
        
        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        self.config = config
        
        # Initialize Accelerate - this handles all distributed setup
        # When using DeepSpeed config file, we should not pass conflicting parameters
        accelerator_kwargs = {
            "log_with": None,  # We'll use our own logging
            "project_dir": config.job.path_local,
            "split_batches": False,
        }
        
        # Only add these parameters if not using DeepSpeed config file
        use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
        if not use_deepspeed:
            # Determine mixed precision setting
            mixed_precision = "no"
            if hasattr(config.trainer, 'mixed_precision') and config.trainer.mixed_precision:
                mixed_precision = "bf16"
            elif hasattr(config.trainer, 'amp') and config.trainer.amp:
                mixed_precision = "bf16"
            
            accelerator_kwargs.update({
                "gradient_accumulation_steps": config.trainer.grad_accum_iter,
                "mixed_precision": mixed_precision,
            })
        
        self.accelerator = Accelerator(**accelerator_kwargs)
        
        # Set up logging and directories
        if self.accelerator.is_main_process:
            os.makedirs(config.job.path_local, exist_ok=True)
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        
        self.accelerator.wait_for_everyone()
        log.init_loguru_file(f"{config.job.path_local}/stdout.log")
        
        if self.accelerator.is_main_process:
            log.info("Config:\n" + config.pretty_print(use_color=True))
        
        misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        
        # Set random seed using Accelerate's utility
        set_seed(config.trainer.seed)
        
        # Initialize cuDNN settings
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        
        # Initialize callbacks
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        
        # Initialize checkpointer
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        
        # Initialize timer
        self.training_timer = misc.TrainingTimer()
        
        # Set up timeout handler
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The main training function using Accelerate.
        
        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Model preparation (but don't move to device - Accelerate will handle this)
        model.on_train_start(self.config.trainer.memory_format)
        
        # Initialize optimizer and scheduler
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        self.callbacks.on_optimizer_init_end()
        
        # Prepare everything with Accelerate
        model, optimizer, dataloader_train, dataloader_val, scheduler = self.accelerator.prepare(
            model, optimizer, dataloader_train, dataloader_val, scheduler
        )
        
        # Load checkpoint and get starting iteration
        iteration = self.checkpointer.load(model, optimizer, scheduler, None)  # No grad_scaler with Accelerate
        
        log.info("Starting training with Accelerate...")
        self.callbacks.on_train_start(model, iteration=iteration)
        
        # Initial validation
        if self.config.trainer.run_validation and iteration == 0 and dataloader_val is not None:
            self.validate(model, dataloader_val, iteration=iteration)
        
        with maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler, \
             maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler:
            
            while iteration < self.config.trainer.max_iter:
                for data_batch in dataloader_train:
                    if iteration >= self.config.trainer.max_iter:
                        break
                    
                    self.callbacks.on_before_dataloading(iteration)
                    self.callbacks.on_after_dataloading(iteration)
                    
                    # Training step with Accelerate
                    loss = self.training_step(model, optimizer, scheduler, data_batch, iteration)
                    
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, {}, loss, iteration=iteration
                    )
                    
                    iteration += 1
                    
                    # Save checkpoint
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, None, iteration=iteration)
                    
                    self.callbacks.on_training_step_end(model, data_batch, {}, loss, iteration=iteration)
                    
                    # Validation
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0 and dataloader_val is not None:
                        self.validate(model, dataloader_val, iteration=iteration)
                    
                    # Reset timeout signal
                    signal.alarm(self.config.trainer.timeout_period)
                    
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
        
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, None, iteration=iteration)
        
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        self.accelerator.wait_for_everyone()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> torch.Tensor:
        """Training step using Accelerate's gradient accumulation.
        
        Args:
            model (torch.nn.Module): The prepared model.
            optimizer (torch.optim.Optimizer): The prepared optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The prepared scheduler.
            data_batch (dict[str, torch.Tensor]): Data batch.
            iteration (int): Current iteration number.
            
        Returns:
            loss (torch.Tensor): The total loss.
        """
        model.train()
        
        self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
        self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
        
        with self.accelerator.accumulate(model):
            self.callbacks.on_before_forward(iteration=iteration)
            
            with self.training_timer("forward"):
                # Call the model's training_step method
                output_batch, loss = model.training_step(data_batch, iteration)
            
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model, loss, iteration=iteration)
            
            with self.training_timer("backward"):
                # Accelerate handles gradient scaling automatically
                self.accelerator.backward(loss)
                
                # Call model's after backward hook
                if hasattr(model, 'module'):  # If wrapped by DDP/FSDP
                    model.module.on_after_backward(iteration)
                else:
                    model.on_after_backward(iteration)
            
            self.callbacks.on_after_backward(model, iteration=iteration)
            
            # Optimizer step (only happens when gradients are accumulated)
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model, optimizer, scheduler, None, iteration=iteration  # No grad_scaler
                )
                
                optimizer.step()
                scheduler.step()
                
                self.callbacks.on_before_zero_grad(model, optimizer, scheduler, iteration=iteration)
                
                # Call model's before zero grad hook
                if hasattr(model, 'module'):
                    model.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                else:
                    model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                
                optimizer.zero_grad()
        
        return loss

    @torch.no_grad()
    def validate(
        self, 
        model: torch.nn.Module, 
        dataloader_val: torch.utils.data.DataLoader, 
        iteration: int = 0
    ) -> None:
        """Validation loop using Accelerate.
        
        Args:
            model (torch.nn.Module): The prepared model.
            dataloader_val (torch.utils.data.DataLoader): The prepared validation dataloader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        
        # Get the underlying model for EMA
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        with ema.ema_scope(unwrapped_model, enabled=unwrapped_model.config.ema.enabled):
            for val_iter, data_batch in enumerate(dataloader_val):
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                
                # Call model's validation_step
                if hasattr(model, 'module'):
                    output_batch, loss = model.module.validation_step(data_batch, iteration)
                else:
                    output_batch, loss = model.validation_step(data_batch, iteration)
                
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        
        self.callbacks.on_validation_end(model, iteration=iteration)


def create_accelerate_compatible_dataloader(dataloader_config, accelerator=None):
    """Create a dataloader compatible with Accelerate by avoiding Megatron parallel state."""
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    
    # Handle None config
    if dataloader_config is None:
        return None
    
    # Create the dataset first
    dataset = instantiate(dataloader_config["dataset"])
    
    # Create sampler without using Megatron parallel state
    sampler = None
    if accelerator is not None and accelerator.num_processes > 1:
        # Use Accelerate's distributed info
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=dataloader_config.get("shuffle", True),
            seed=0,
        )
    elif dist.is_available() and dist.is_initialized():
        # Fallback to torch.distributed
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=dataloader_config.get("shuffle", True),
            seed=0,
        )
    
    # Create dataloader
    # Filter out config-specific keys that aren't valid DataLoader arguments
    config_keys_to_exclude = ["dataset", "sampler", "_target_", "_partial_", "_recursive_"]
    dataloader_kwargs = {k: v for k, v in dataloader_config.items() if k not in config_keys_to_exclude}
    
    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler
        dataloader_kwargs.pop("shuffle", None)  # Can't use shuffle with sampler
    
    return DataLoader(dataset, **dataloader_kwargs)


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    """Launch training with Accelerate."""
    # Check for FSDP/DeepSpeed conflict and adjust config if needed BEFORE freezing
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
    if use_deepspeed:
        # Disable FSDP trainer setting
        if hasattr(config.trainer, 'distributed_parallelism') and config.trainer.distributed_parallelism == "fsdp":
            log.warning("Detected FSDP trainer config with DeepSpeed. Switching distributed_parallelism to 'ddp' for compatibility...")
            config.trainer.distributed_parallelism = "ddp"
        
        # Disable FSDP in model config
        if hasattr(config.model, 'config') and hasattr(config.model.config, 'fsdp_shard_size') and config.model.config.fsdp_shard_size != 0:
            log.warning(f"Detected FSDP model config (fsdp_shard_size={config.model.config.fsdp_shard_size}) with DeepSpeed. Disabling FSDP in model...")
            config.model.config.fsdp_shard_size = 0
    
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    
    # Create trainer (this initializes Accelerate and distributed environment)
    trainer = AccelerateTrainer(config)
    
    # Create the model
    model = instantiate(config.model)
    
    # Create dataloaders after Accelerate is initialized
    # We need to handle this specially because the original configs use Megatron parallel state
    try:
        dataloader_train = instantiate(config.dataloader_train) if config.dataloader_train is not None else None
        dataloader_val = instantiate(config.dataloader_val) if config.dataloader_val is not None else None
    except (AssertionError, AttributeError) as e:
        if "data parallel group is not initialized" in str(e) or "parallel_state" in str(e):
            log.warning("Original dataloader config uses Megatron parallel state. Creating Accelerate-compatible dataloaders...")
            dataloader_train = create_accelerate_compatible_dataloader(config.dataloader_train, trainer.accelerator)
            dataloader_val = create_accelerate_compatible_dataloader(config.dataloader_val, trainer.accelerator)
        else:
            raise e
    
    # Validation dataloader is optional
    if dataloader_train is None:
        raise ValueError("Training dataloader cannot be None")
    
    if dataloader_val is None:
        log.warning("Validation dataloader is None. Validation will be skipped.")
    
    # Start training
    trainer.train(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    # Usage: accelerate launch --config_file deepspeed_config.yaml scripts/train_accel.py --config=cosmos_predict2/configs/base/config.py -- experiments=predict2_video2world_training_2b_cosmos_nemo_assets

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training with Accelerate")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    
    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args) 