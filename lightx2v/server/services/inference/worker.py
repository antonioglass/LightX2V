import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import torch
from loguru import logger

from lightx2v.infer import init_runner
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import set_config, set_parallel_config

from ..distributed_utils import DistributedManager


class TorchrunInferenceWorker:
    def __init__(self):
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.runner = None
        self.dist_manager = DistributedManager()
        self.processing = False
        self.lora_dir = None
        self.is_moe_runner = False
        self.current_lora_name = None
        self.current_lora_strength = None
        self.current_high_lora_name = None
        self.current_high_lora_strength = None
        self.current_low_lora_name = None
        self.current_low_lora_strength = None

    def init(self, args) -> bool:
        try:
            if self.world_size > 1:
                if not self.dist_manager.init_process_group():
                    raise RuntimeError("Failed to initialize distributed process group")
            else:
                self.dist_manager.rank = 0
                self.dist_manager.world_size = 1
                self.dist_manager.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.dist_manager.is_initialized = False

            self.lora_dir = getattr(args, "lora_dir", None)
            if self.lora_dir:
                self.lora_dir = Path(self.lora_dir)
                if not self.lora_dir.exists():
                    logger.warning(f"LoRA directory does not exist: {self.lora_dir}")
                    self.lora_dir = None
                else:
                    logger.info(f"LoRA directory set to: {self.lora_dir}")

            config = set_config(args)

            if config["parallel"]:
                set_parallel_config(config)

            if self.rank == 0:
                logger.info(f"Config:\n {config}")

            self.runner = init_runner(config)
            logger.info(f"Rank {self.rank}/{self.world_size - 1} initialization completed")

            # Detect MoE runner
            self.is_moe_runner = "moe" in config.get("model_cls", "")

            self.input_info = init_empty_input_info(args.task)

            return True

        except Exception as e:
            logger.exception(f"Rank {self.rank} initialization failed: {str(e)}")
            return False

    async def process_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        has_error = False
        error_msg = ""

        try:
            if self.world_size > 1 and self.rank == 0:
                task_data = self.dist_manager.broadcast_task_data(task_data)

            # Handle dynamic LoRA loading
            lora_name = task_data.pop("lora_name", None)
            lora_strength = task_data.pop("lora_strength", 1.0)
            high_lora_name = task_data.pop("high_lora_name", None)
            high_lora_strength = task_data.pop("high_lora_strength", 1.0)
            low_lora_name = task_data.pop("low_lora_name", None)
            low_lora_strength = task_data.pop("low_lora_strength", 1.0)

            if self.is_moe_runner:
                self.switch_moe_lora(high_lora_name, high_lora_strength, low_lora_name, low_lora_strength)
            elif self.lora_dir:
                self.switch_lora(lora_name, lora_strength)

            task_data["task"] = self.runner.config["task"]
            task_data["return_result_tensor"] = False
            task_data["negative_prompt"] = task_data.get("negative_prompt", "")

            target_fps = task_data.pop("target_fps", None)
            if target_fps is not None:
                vfi_cfg = self.runner.config.get("video_frame_interpolation")
                if vfi_cfg:
                    task_data["video_frame_interpolation"] = {**vfi_cfg, "target_fps": target_fps}
                else:
                    logger.warning(f"Target FPS {target_fps} is set, but video frame interpolation is not configured")

            update_input_info_from_dict(self.input_info, task_data)

            self.runner.set_config(task_data)
            self.runner.run_pipeline(self.input_info)

            await asyncio.sleep(0)

        except Exception as e:
            has_error = True
            error_msg = str(e)
            logger.exception(f"Rank {self.rank} inference failed: {error_msg}")

        if self.world_size > 1:
            self.dist_manager.barrier()

        if self.rank == 0:
            if has_error:
                return {
                    "task_id": task_data.get("task_id", "unknown"),
                    "status": "failed",
                    "error": error_msg,
                    "message": f"Inference failed: {error_msg}",
                }
            else:
                return {
                    "task_id": task_data["task_id"],
                    "status": "success",
                    "save_result_path": task_data["save_result_path"],
                    "message": "Inference completed",
                }
        else:
            return None

    def switch_lora(self, lora_name: str, lora_strength: float):
        """Switch LoRA for single-model (non-MoE) runners."""
        try:
            if lora_name == self.current_lora_name and lora_strength == self.current_lora_strength:
                return

            model = self.runner.model
            # Always remove old LoRA first (safe even if none was registered)
            model._remove_lora()

            if lora_name is not None:
                lora_path = self._lora_path(lora_name)
                if lora_path is None:
                    logger.warning(f"LoRA file not found: {lora_name}")
                    self.current_lora_name = None
                    self.current_lora_strength = None
                    return
                logger.info(f"Applying LoRA: {lora_name} (strength={lora_strength})")
                model._register_lora(lora_path, lora_strength)
            else:
                logger.info("LoRA disabled")

            self.current_lora_name = lora_name
            self.current_lora_strength = lora_strength

        except Exception as e:
            logger.error(f"Failed to handle LoRA switching: {e}")
            raise

    def switch_moe_lora(self, high_lora_name, high_lora_strength,
                         low_lora_name, low_lora_strength):
        """Switch LoRA weights for MoE models with separate high/low noise models."""
        try:
            high_changed = (high_lora_name != self.current_high_lora_name
                            or high_lora_strength != self.current_high_lora_strength)
            low_changed = (low_lora_name != self.current_low_lora_name
                           or low_lora_strength != self.current_low_lora_strength)

            if not high_changed and not low_changed:
                return

            high_model = self.runner.model.model[0]
            low_model = self.runner.model.model[1]

            if high_changed:
                self._switch_sub_model_lora(high_model, "high-noise", high_lora_name, high_lora_strength)
                self.current_high_lora_name = high_lora_name
                self.current_high_lora_strength = high_lora_strength

            if low_changed:
                self._switch_sub_model_lora(low_model, "low-noise", low_lora_name, low_lora_strength)
                self.current_low_lora_name = low_lora_name
                self.current_low_lora_strength = low_lora_strength

            logger.info("MoE LoRA switch completed")

        except Exception as e:
            logger.error(f"Failed to handle MoE LoRA switching: {e}")
            raise

    def _switch_sub_model_lora(self, sub_model, label, lora_name, lora_strength):
        """Switch or disable LoRA on a single sub-model.

        Fully removes old LoRA (if any) and registers a new one from scratch.
        register_lora/remove_lora recurse through all sub-modules including
        the offload CUDA buffers, so no separate handling is needed.
        """
        if sub_model is None:
            return

        # Always remove old LoRA first (safe even if none was registered)
        sub_model._remove_lora()

        if lora_name is not None:
            lora_path = self._lora_path(lora_name)
            if lora_path is None:
                logger.warning(f"{label} LoRA file not found: {lora_name}")
                self._invalidate_offload_buffer(sub_model)
                return
            logger.info(f"Switching {label} LoRA to: {lora_name} (strength={lora_strength})")
            sub_model._register_lora(lora_path, lora_strength)
        else:
            logger.info(f"Disabled {label} LoRA")

        self._invalidate_offload_buffer(sub_model)

    def _invalidate_offload_buffer(self, sub_model):
        """Force the offload manager to re-copy block 0 on next inference."""
        infer = getattr(sub_model, "transformer_infer", None)
        if infer and hasattr(infer, "offload_manager"):
            infer.offload_manager.need_init_first_buffer = True

    def _lora_path(self, lora_name: str) -> str:
        if not self.lora_dir:
            return None
        lora_file = self.lora_dir / lora_name
        if lora_file.exists():
            return str(lora_file)
        return None

    async def worker_loop(self):
        while True:
            task_data = None
            try:
                task_data = self.dist_manager.broadcast_task_data()
                if task_data is None:
                    logger.info(f"Rank {self.rank} received stop signal")
                    break

                await self.process_request(task_data)

            except Exception as e:
                error_str = str(e)
                if "Connection closed by peer" in error_str or "Connection reset by peer" in error_str:
                    logger.info(f"Rank {self.rank} detected master process shutdown, exiting worker loop")
                    break
                logger.error(f"Rank {self.rank} worker loop error: {error_str}")
                if self.world_size > 1 and task_data is not None:
                    try:
                        self.dist_manager.barrier()
                    except Exception as barrier_error:
                        logger.warning(f"Rank {self.rank} barrier failed, exiting: {barrier_error}")
                        break
                continue

    def cleanup(self):
        self.dist_manager.cleanup()
