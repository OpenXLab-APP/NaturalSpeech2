import json
import os
import random
import shutil
import torch
import time
from abc import abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import re
from utils.util import Logger, ValueWindow

import accelerate
import json5
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import ConcatDataset, DataLoader
from accelerate import DistributedDataParallelKwargs

from models.base.base_sampler import build_samplers


class TTSTrainer:
    r"""The base trainer for all TTS models. It inherits from BaseTrainer and implements
    ``build_criterion``, ``_build_dataset`` and ``_build_singer_lut`` methods. You can inherit from this
    class, and implement ``_build_model``, ``_forward_step``.
    """

    def __init__(self, args=None, cfg=None):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        self.time_window = ValueWindow(50)

        if self.accelerator.is_main_process:
            # Log some info
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            (
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.train_dataloader,
                self.valid_dataloader,
            )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # TODO: Resume from ckpt need test/debug
        with self.accelerator.main_process_first():
            if args.resume:
                if self.accelerator.is_main_process:
                    self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = self._load_model(
                    self.checkpoint_dir,
                    args.checkpoint_path,
                    resume_type=args.resume_type,
                )
                end = time.monotonic_ns()
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # Only for TTS tasks
        self.task_type = "TTS"
        if self.accelerator.is_main_process:
            self.logger.info("Task type: {}".format(self.task_type))

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    ### Following are methods only for TTS tasks ###
    # TODO: LEGACY CODE, NEED TO BE REFACTORED
    def _build_dataset(self):
        pass

    def _build_criterion():
        pass

    def _build_model(self):
        pass

    def _build_dataloader(self):
        pass

    def _build_optimizer(self):
        pass

    def _build_scheduler(self):
        pass

    def _load_model(self, checkpoint_dir, checkpoint_path=None, resume_type="resume"):
        """Load model from checkpoint. If a folder is given, it will
        load the latest checkpoint in checkpoint_dir. If a path is given
        it will load the checkpoint specified by checkpoint_path.
        **Only use this method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
            ls.sort(key=lambda x: int(x.split("_")[-3].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
        if resume_type == "resume":
            self.accelerator.load_state(checkpoint_path)
        elif resume_type == "finetune":
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            if self.accelerator.is_main_process:
                self.logger.info("Load model weights for finetune SUCCESS!")
        else:
            raise ValueError("Unsupported resume type: {}".format(resume_type))
        self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1]) + 1
        self.step = int(checkpoint_path.split("_")[-2].split("-")[-1]) + 1
        return checkpoint_path

    ### THIS IS MAIN ENTRY ###
    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)

        # self.optimizer.zero_grad()

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            valid_total_loss, valid_losses = self._valid_epoch()
            if isinstance(valid_losses, dict):
                for key, loss in valid_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # Check if hit save_checkpoint_stride and run_eval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                hit_dix = []
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        hit_dix.append(i)
                        run_eval |= self.run_eval[i]

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
                print("save state......")
                self.accelerator.save_state(path)
                print("finish saving state......")
                json.dump(
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                # Remove old checkpoints
                to_remove = []
                for idx in hit_dix:
                    self.checkpoints_path[idx].append(path)
                    while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                        to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

                # Search conflicts
                total = set()
                for i in self.checkpoints_path:
                    total |= set(i)
                do_remove = set()
                for idx, path in to_remove[::-1]:
                    if path in total:
                        self.checkpoints_path[idx].insert(0, path)
                    else:
                        do_remove.add(path)

                # Remove old checkpoints
                for path in do_remove:
                    shutil.rmtree(path, ignore_errors=True)
                    if self.accelerator.is_main_process:
                        self.logger.debug(f"Remove old checkpoint: {path}")

            self.accelerator.wait_for_everyone()
            if run_eval:
                # TODO: run evaluation
                pass

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
            )
        self.accelerator.end_training()

    ### Following are methods that can be used directly in child classes ###
    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1

            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (1 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = dict()

        for batch in self.valid_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss = total_loss
            for key, value in valid_losses.items():
                epoch_losses[key] = value

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def _train_step(self):
        pass

    def _valid_step(self):
        pass

    def _inference(self):
        pass

    def _set_random_seed(self, seed):
        """Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _check_nan(self, loss):
        if torch.any(torch.isnan(loss)):
            if self.accelerator.is_main_process:
                self.logger.fatal("Fatal Error: NaN!")
                self.logger.error("loss = {:.6f}".format(loss.item()), in_order=True)

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            if self.accelerator.is_main_process:
                self.logger.fatal("Invalid gradient_accumulation_step value!")
                self.logger.error(
                    f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
                )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )

    @staticmethod
    def _count_parameters(model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    def _is_valid_pattern(self, directory_name):
        directory_name = str(directory_name)
        pattern = r"^epoch-\d{4}_step-\d{7}_loss-\d{1}\.\d{6}"
        return re.match(pattern, directory_name) is not None

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            if self.accelerator.is_main_process:
                self.logger.fatal("Invalid gradient_accumulation_step value!")
                self.logger.error(
                    f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
                )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(
                    str(key).split("/")[-1] + "=" + str(round(float(losses[key]), 5))
                )
        self.logger.info(", ".join(message))
