"""Main script for keypose optimization."""

import os
from pathlib import Path
import random
from typing import Tuple, Optional

import numpy as np
import tap
import torch
import torch.distributed as dist
from torch.nn import functional as F

from datasets import RLBenchDataset
from engine import BaseTrainTester
from utils.utils_without_rlbench import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)


class Arguments(tap.Tap):
    # master_port: str = '29500'
    local_rank: int
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None

    # Training and validation datasets
    dataset: Path
    valset: Path

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    logger: Optional[str] = "tensorboard"  # One of "tensorboard", None
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    n_gpus: int = 0
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    train_iters: int = 200_000
    max_episode_length: int = 5  # -1 for no limit

    # Toggle to switch between our models
    model: str = "diffusion"  # one of "diffusion", "regression"

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling
    point_cloud_rotate_yaw_range: float = 0.0  # in degrees, 0.0 for no rot

    # Model
    action_dim: int = 7
    backbone: str = "clip"  # one of "resnet", "clip"
    num_sampling_level: int = 3
    embedding_dim: int = 60
    num_query_cross_attn_layers: int = 8
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    use_rgb: int = 1
    feat_scales_to_use: int = 1
    attn_rounds: int = 1
    diffusion_head: str = "simple"


class TrainTester(BaseTrainTester):
    """Train/test a keypose optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        instruction = load_instructions(
            self.args.instructions,
            tasks=self.args.tasks,
            variations=self.args.variations
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]

        # Initialize datasets with arguments
        train_dataset = RLBenchDataset(
            root=self.args.dataset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cache_size=self.args.cache_size,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            point_cloud_rotate_yaw_range=self.args.point_cloud_rotate_yaw_range,
            return_low_lvl_trajectory=False,
            dense_interpolation=False,
            interpolation_length=0,
            action_dim=self.args.action_dim,
            predict_short=False
        )
        test_dataset = RLBenchDataset(
            root=self.args.valset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cache_size=self.args.cache_size_val,
            cameras=self.args.cameras,
            training=False,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            point_cloud_rotate_yaw_range=self.args.point_cloud_rotate_yaw_range,
            return_low_lvl_trajectory=False,
            dense_interpolation=False,
            interpolation_length=0,
            action_dim=self.args.action_dim,
            predict_short=False
        )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Select model class
        if self.args.model == "diffusion":
            from model import DiffusionHLPlanner
            model_class = DiffusionHLPlanner
        elif self.args.model == "regression":
            from model import TrajectoryHLRegressor
            model_class = TrajectoryHLRegressor

        # Initialize model with arguments
        _model = model_class(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            output_dim=self.args.action_dim,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            num_query_cross_attn_layers=self.args.num_query_cross_attn_layers,
            num_sampling_level=self.args.num_sampling_level,
            use_instruction=bool(self.args.use_instruction),
            use_rgb=bool(self.args.use_rgb),
            feat_scales_to_use=self.args.feat_scales_to_use,
            attn_rounds=self.args.attn_rounds,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            diffusion_head=self.args.diffusion_head
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    @staticmethod
    def get_criterion():
        return KeyposeCriterion()

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        # Forward pass
        out = model(
            sample["action"],
            sample["rgbs"],
            sample["pcds"],
            sample["instr"],
            sample["curr_gripper"]
        )

        # Backward pass
        loss = criterion.compute_loss(out)
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            action = model(
                sample["action"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"].to(device),
                sample["curr_gripper"].to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["action"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            for key, val in values.items():
                self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/action_mse', None)


def keypose_collate_fn(batch):
    # Unfold multi-step demos to form a longer batch
    keys = ["rgbs", "pcds", "curr_gripper", "action", "instr"]
    ret_dict = {key: torch.cat([item[key] for item in batch]) for key in keys}

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict


class KeyposeCriterion:

    def __init__(
        self,
        pos_loss_coeff=1.0,
        rot_loss_coeff=10.0,
        grp_loss_coeff=1.0
    ):
        self.pos_loss_coeff = pos_loss_coeff
        self.rot_loss_coeff = rot_loss_coeff
        self.grp_loss_coeff = grp_loss_coeff

    def compute_loss(self, pred, gt=None, is_loss=True):
        if not is_loss:
            assert gt is not None
            return self.compute_metrics(pred, gt)[0]['action_mse']
        return pred

    def compute_metrics(self, pred, gt):
        # pred/gt are (B, 7)
        pos_loss = F.mse_loss(pred[..., :3], gt[..., :3])
        rot_loss = F.mse_loss(pred[..., 3:7], gt[..., 3:7])
        grp_loss = 0  # F.mse_loss(pred[..., 7:8], gt[..., 7:8])
        ret_dict = {
            "action_mse": F.mse_loss(pred, gt),
            "pos_mse": pos_loss * self.pos_loss_coeff,
            "rot_mse": rot_loss * self.rot_loss_coeff,
            # "grp_mse": grp_loss * self.grp_loss_coeff
        }
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        batch_dict = {
            'pos_l2': pos_l2,
            'pos_acc_001': (pos_l2 < 0.01).float(),
            'rot_l1': quat_l1,
            'rot_acc_005': (quat_l1 < 0.05).float(),
            'rot_acc_0025': (quat_l1 < 0.025).float(),
            # 'gripper_acc': ((pred[..., 7] > 0.5) == gt[..., 7].bool()).float()
        }
        ret_dict.update({key: val.mean() for key, val in batch_dict.items()})

        return ret_dict, batch_dict


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=0.04
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print("Args n_gpus:", args.n_gpus)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=keypose_collate_fn)
