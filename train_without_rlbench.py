import random
from typing import List, Tuple, Dict, Optional, Any
import os
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm, trange
from filelock import FileLock
import tap
from network import Hiveformer
from utils_without_rlbench import (
    LossAndMetrics,
    load_instructions,
    # RLBenchEnv,
    count_parameters,
    load_episodes,
    get_max_episode_length,
    # Actioner,
)
from dataset import RLBenchDataset

# from baseline.baseline import Baseline


class Arguments(tap.Tap):
    accumulate_grad_batches: int = 1
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    checkpoint: Optional[Path] = None
    checkpoint_period: int = 10
    dataset: List[Path]
    devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    xp: Path = Path(__file__).parent / "xp"
    valset: Optional[Tuple[Path, ...]] = None
    name: str = "hiveformer"
    arch: str = "mct"
    num_workers: int = 5 * len(devices)
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    cache_size: int = 100
    seed: int = 2

    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)

    # Train
    batch_size: int = 32 * len(devices)
    lr: float = 0.001
    val_freq: int = 200
    train_iters: int = 100_000 // len(devices)
    jitter: bool = False

    # tests
    headless: bool = False
    output: Path = Path(__file__).parent / "records.txt"

    # model
    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1


def training(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loaders,
    checkpointer,
    loss_and_metrics,
    args: Arguments,
    writer: SummaryWriter,
):
    iter_loader = iter(train_loader)

    with trange(args.train_iters) as tbar:
        for step_id in tbar:
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            if step_id % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            pred = model(
                sample["rgbs"],
                sample["pcds"],
                sample["padding_mask"],
                sample["instr"],
                sample["gripper"],
            )

            train_losses = loss_and_metrics.compute_loss(pred, sample)
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore

            for n, l in train_losses.items():
                writer.add_scalar(f"train-loss/{n}", l, step_id)

            writer.add_scalar(f"lr/", args.lr, step_id)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                writer.add_scalar(f"train-metrics/{n}", l, step_id)

            train_losses["total"].backward()  # type: ignore

            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if (step_id + 1) % args.val_freq == 0:
                if val_loaders is not None:
                    val_metrics = validation_step(
                        step_id,
                        val_loaders,
                        model,
                        writer,
                        loss_and_metrics,
                    )
                    model.train()
                else:
                    val_metrics = {}
                checkpointer(val_metrics)

            tbar.set_postfix(l=float(train_losses["total"]))


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.xp / args.name
    version = int(os.environ.get("SLURM_JOBID", 0))
    while (log_dir / f"version{version}").is_dir():
        version += 1
    return log_dir / f"version{version}"


class CheckpointCallback:
    def __init__(
        self,
        name: str,
        log_dir: Path,
        state_dict: Any,
        minimizing: bool = True,
        checkpoint_period: int = 200,
    ):
        self._name = name
        self._minimizing = minimizing
        self._best = float("inf") if minimizing else -float("inf")
        self._log_dir = log_dir
        self._checkpoint_period = checkpoint_period
        self._step = 0
        self._state_dict = state_dict

    def __call__(self, metrics: Dict[str, torch.Tensor]):
        self._step += 1
        if self._step % self._checkpoint_period != 0:
            return

        value = int(metrics.get(self._name, 0))
        dest = self._log_dir / f"model.step={self._step}-value={value}.pth"
        torch.save(self._state_dict, dest)

        if (self._minimizing and self._best > value) or (
            not self._minimizing and self._best < value
        ):
            best = self._log_dir / "best.pth"
            best.unlink(missing_ok=True)
            best.symlink_to(dest.resolve())
            self._best = value


@torch.no_grad()
def validation_step(
    step_id: int,
    val_loaders: List[DataLoader],
    model,
    writer,
    loss_and_metrics,
    val_iters: int = 5,
):
    values = {}
    device = next(model.parameters()).device
    model.eval()

    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            pred = model(
                sample["rgbs"],
                sample["pcds"],
                sample["padding_mask"],
                sample["instr"],
                sample["gripper"],
            )

            losses: Dict[str, torch.Tensor] = loss_and_metrics.compute_loss(pred, sample)
            losses["total"] = torch.stack(list(losses.values())).sum()

            for n, l in losses.items():
                key = f"val-loss-{val_id}/{n}"
                writer.add_scalar(key, l, step_id + i)
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            writer.add_scalar(f"lr/", args.lr, step_id + i)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                key = f"val-metrics-{val_id}/{n}"
                writer.add_scalar(key, l, step_id + i)
                if key not in metrics:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        key = f"val-loss-{val_id}/total"
        print(f"Validation Loss {val_id}: {values[key].mean():.05f}")
        key = f"val-metrics-{val_id}/position"
        print(f"Validation Position {val_id}: {values[key].mean():.05f}")

    return values


def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key] for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }


def get_train_loader(args: Arguments) -> DataLoader:
    instruction = load_instructions(
        args.instructions, tasks=args.tasks, variations=args.variations
    )

    if instruction is None:
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    print(f"Valset has {len(taskvar)} taskvars")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    dataset = RLBenchDataset(
        root=args.dataset,
        taskvar=taskvar,
        instructions=instruction,
        max_episode_length=max_episode_length,
        max_episodes_per_taskvar=args.max_episodes_per_taskvar,
        cache_size=args.cache_size,
        num_iters=args.train_iters,
        cameras=args.cameras,  # type: ignore
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return loader


def get_val_loaders(args: Arguments) -> Optional[List[DataLoader]]:
    if args.valset is None:
        return None

    instruction = load_instructions(
        args.instructions, tasks=args.tasks, variations=args.variations
    )

    if instruction is None:
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    print(f"Valset has {len(taskvar)} taskvars")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    loaders = []

    for valset in args.valset:
        dataset = RLBenchDataset(
            root=valset,
            taskvar=taskvar,
            instructions=instruction,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=args.max_episodes_per_taskvar,
            cache_size=args.cache_size,
            training=False,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        loaders.append(loader)

    print(len(loaders), "validation loaders")

    return loaders


def get_model(args: Arguments) -> Tuple[optim.Optimizer, Hiveformer]:
    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    _model = Hiveformer(
        depth=args.depth,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        instr_size=args.instr_size,
        mask_obs_prob=args.mask_obs_prob,
        max_episode_length=max_episode_length,
        num_layers=args.num_layers,
    )
    # _model = Baseline()

    devices = [torch.device(d) for d in args.devices]
    model = _model.to(devices[0])
    if args.devices[0] != "cpu":
        assert all("cuda" in d for d in args.devices)
        model = torch.nn.DataParallel(model, device_ids=devices)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": args.lr},
        {"params": [], "weight_decay": 5e-4, "lr": args.lr},
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in _model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)  # type: ignore
        else:
            optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
    optimizer: optim.Optimizer = optim.AdamW(optimizer_grouped_parameters)

    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        _model.load_state_dict(model_dict["weight"])
        optimizer.load_state_dict(model_dict["optimizer"])

    print("Number of parameters:")
    model_params = count_parameters(_model)
    print("- model", model_params)
    print("Total", model_params)

    return optimizer, model


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    args.save(str(log_dir / "hparams.json"))
    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    optimizer, model = get_model(args)

    loss_and_metrics = LossAndMetrics()

    # training episode
    model_dict = {
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-metrics/position",
        log_dir,
        model_dict,
        minimizing=False,
        checkpoint_period=args.checkpoint_period,
    )
    model.train()

    val_loaders = get_val_loaders(args)

    if args.train_iters > 0:
        train_loader = get_train_loader(args)
        training(
            model,
            optimizer,
            train_loader,
            val_loaders,
            checkpointer,
            loss_and_metrics,
            args,
            writer,
        )

    if val_loaders is not None:
        val_metrics = validation_step(
            args.train_iters,
            val_loaders,
            model,
            writer,
            loss_and_metrics,
            val_iters=-1,
        )

    # last checkpoint
    checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    torch.save(model_dict, checkpoint)

    # evaluation
    # model.eval()
    #
    # env = RLBenchEnv(
    #     data_path="",
    #     apply_rgb=True,
    #     apply_pc=True,
    #     apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
    #     headless=args.headless,
    # )
    #
    # instruction = load_instructions(args.instructions)
    # if instruction is None:
    #     raise NotImplementedError()
    #
    # actioner = Actioner(model=model, instructions=instruction)
    # max_eps_dict = load_episodes()["max_episode_length"]
    # for task_str in args.tasks:
    #     for variation in args.variations:
    #         success_rate = env.evaluate(
    #             task_str,
    #             actioner=actioner,
    #             max_episodes=max_eps_dict.get(task_str, 6),
    #             variation=variation,
    #             num_demos=500,
    #             demos=None,
    #             log_dir=log_dir,
    #             max_tries=args.max_tries,
    #             record_videos=False
    #         )
    #
    #         print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
    #
    #         with FileLock(args.output.parent / f"{args.output.name}.lock"):
    #             with open(args.output, "a") as oid:
    #                 oid.write(
    #                     f"{task_str}-{variation}, na, seed={args.seed}, {success_rate}\n"
    #                 )
