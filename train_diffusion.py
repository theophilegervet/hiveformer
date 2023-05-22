import random
import os
from collections import defaultdict
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import trange
import wandb

from utils.utils_without_rlbench import (
    load_instructions,
    count_parameters,
    get_max_episode_length,
    get_gripper_loc_bounds
)
from dataset import RLBenchDataset
from model.diffusion_planner.diffusion_model import DiffusionPlanner
from train import Arguments, get_log_dir, CheckpointCallback


def training(
    model,
    optimizer,
    train_loader,
    val_loaders,
    checkpointer,
    args,
    writer=None
):
    iter_loader = iter(train_loader)

    aggregated_losses = defaultdict(list)

    with trange(args.train_iters) as tbar:
        for step_id in tbar:
            try:
                sample = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            if step_id % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            loss = model(
                sample["trajectory"],
                sample["trajectory_mask"],
                sample["rgbs"],
                sample["pcds"],
                sample["instr"],
                sample["curr_gripper"],
                sample["action"]
            )

            loss.backward()
            aggregated_losses["mse"].append(loss.item())

            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if args.logger == "wandb":
                wandb.log(
                    {
                        "lr": args.lr,
                        **{f"train-loss/{n}": torch.mean(torch.stack(l)) for n, l in aggregated_losses.items()}
                    },
                    step=step_id
                )

            if (step_id + 1) % args.val_freq == 0:
                if args.logger == "tensorboard":
                    writer.add_scalar(f"lr/", args.lr, step_id)
                    for n, l in aggregated_losses.items():
                        writer.add_scalar(
                            f"train-loss/{n}",
                            torch.mean(torch.as_tensor(l)),
                            step_id
                        )

                aggregated_losses = defaultdict(list)

                if val_loaders is not None:
                    val_metrics = validation_step(
                        step_id,
                        val_loaders,
                        model,
                        args,
                        writer
                    )
                    model.train()
                else:
                    val_metrics = {}
                checkpointer({k: v.mean() for k, v in val_metrics.items()})


@torch.no_grad()
def validation_step(
    step_id,
    val_loaders,
    model,
    args,
    writer=None,
    val_iters=10
):
    values = {}
    device = next(model.parameters()).device
    model.eval()

    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            action = model.module.predict_action(
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"].to(device),
                sample["curr_gripper"].to(device),
                sample["action"].to(device)
            )

            losses = {"mse": F.mse_loss(action, sample["trajectory"].to(device))}

            for n, l in losses.items():
                key = f"val-loss-{val_id}/{n}"
                if args.logger == "tensorboard":
                    writer.add_scalar(key, l, step_id + i)
                elif args.logger == "wandb":
                    wandb.log({key: l}, step=step_id + i)
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            if args.logger == "tensorboard":
                writer.add_scalar(f"lr/", args.lr, step_id + i)
            elif args.logger == "wandb":
                wandb.log({"lr": args.lr}, step=step_id + i)

        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value.mean():.03f}")

    return values


def collate_fn(batch):
    # Dynamic padding for trajectory
    max_len = max(batch[b]['trajectory_len'].max() for b in range(len(batch)))
    for item in batch:
        h, n, c = item['trajectory'].shape
        traj = torch.zeros(h, max_len, c)
        traj[:, :n] = item['trajectory']
        item['trajectory'] = traj

    # Use default collate to batch, then fill missing keys
    ret_dict = {
        key: default_collate([item[key] for item in batch])
        if batch[0][key] is not None
        else None
        for key in batch[0].keys()
    }

    # Unfold multi-step demos to form a longer batch
    keys = [
        "trajectory", "trajectory_len", "rgbs", "pcds",
        "curr_gripper", "action"
    ]
    _mask = ret_dict["padding_mask"]
    ret_dict["instr"] = ret_dict["instr"][:, None].repeat(
        1, ret_dict["rgbs"].size(1), 1, 1
    )[_mask]
    ret_dict["task_id"] = ret_dict["task_id"][:, None].repeat(
        1, ret_dict["rgbs"].size(1)
    )[_mask]
    for key in keys:
        ret_dict[key] = ret_dict[key][_mask]

    # Trajectory mask
    trajectory_mask = torch.zeros(ret_dict['trajectory'].shape[:-1])
    for i, len_ in enumerate(ret_dict['trajectory_len']):
        trajectory_mask[i, len_:] = 1
    ret_dict["trajectory_mask"] = trajectory_mask.bool()

    return ret_dict


def get_train_loader(args, gripper_loc_bounds):
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

    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    if args.max_episode_length >= 0:
        max_episode_length = min(args.max_episode_length, max_episode_length)

    dataset = RLBenchDataset(
        root=args.dataset,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        taskvar=taskvar,
        instructions=instruction,
        max_episode_length=max_episode_length,
        max_episodes_per_task=args.max_episodes_per_task,
        cache_size=args.cache_size,
        num_iters=args.train_iters,
        cameras=args.cameras,
        image_rescale=tuple(float(x) for x in args.image_rescale.split(",")),
        point_cloud_rotate_yaw_range=args.point_cloud_rotate_yaw_range,
        gripper_loc_bounds=gripper_loc_bounds,
        return_low_lvl_trajectory=True
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return loader


def get_val_loaders(args, gripper_loc_bounds):
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

    max_episode_length = get_max_episode_length(args.tasks, args.variations)
    if args.max_episode_length >= 0:
        max_episode_length = min(args.max_episode_length, max_episode_length)

    loaders = []

    for valset in args.valset:
        dataset = RLBenchDataset(
            root=valset,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            taskvar=taskvar,
            instructions=instruction,
            max_episode_length=max_episode_length,
            max_episodes_per_task=args.max_episodes_per_task,
            cache_size=args.cache_size_val,
            cameras=args.cameras,
            training=False,
            image_rescale=tuple(float(x) for x in args.image_rescale.split(",")),
            point_cloud_rotate_yaw_range=args.point_cloud_rotate_yaw_range,
            gripper_loc_bounds=gripper_loc_bounds,
            return_low_lvl_trajectory=True
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size_val,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        loaders.append(loader)

    return loaders


def get_model(args):
    _model = DiffusionPlanner(
        backbone=args.backbone,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        embedding_dim=args.embedding_dim,
        num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
        num_sampling_level=args.num_sampling_level,
        use_instruction=bool(args.use_instruction),
        positional_features=args.positional_features
    )

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
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)
    optimizer = optim.AdamW(optimizer_grouped_parameters)

    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model_dict_weight = {}
        for key in model_dict["weight"]:
            _key = key[7:]
            model_dict_weight[_key] = model_dict["weight"][key]
        _model.load_state_dict(model_dict_weight)
        optimizer.load_state_dict(model_dict["optimizer"])

    model_params = count_parameters(_model)
    print("Model parameters:", model_params)

    return optimizer, model


if __name__ == "__main__":
    args = Arguments().parse_args()

    # Force original HiveFormer parameters
    if args.model == "original":
        assert args.image_size == "128,128"
        args.position_loss = "mse"
        args.position_loss_coeff = 3.0
        args.rotation_loss_coeff = 4.0
        args.batch_size = 32
        args.train_iters = 100_000

    assert args.batch_size % len(args.devices) == 0

    print()
    print("Arguments:")
    print(args)

    print()
    print("-" * 100)
    print()

    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(log_dir / "hparams.json"))

    if args.logger == "tensorboard":
        writer = SummaryWriter(log_dir=log_dir)
    elif args.logger == "wandb":
        wandb.init(project="analogical_manipulation")
        wandb.run.name = str(log_dir).split("/")[-1]
        wandb.config.update(args.__dict__)
        writer = None
    else:
        writer = None

    print("Logging:", log_dir)
    print("Args devices:", args.devices)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_bounds_buffer
    )

    optimizer, model = get_model(args)

    print()
    print("-" * 100)
    print()

    model_dict = {
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-loss-0/mse",
        log_dir,
        model_dict,
        val_freq=args.val_freq,
        minimizing=True,
        checkpoint_freq=args.checkpoint_freq,
    )
    model.train()

    val_loaders = get_val_loaders(args, gripper_loc_bounds)

    if args.train_iters > 0:
        train_loader = get_train_loader(args, gripper_loc_bounds)
        training(
            model,
            optimizer,
            train_loader,
            val_loaders,
            checkpointer,
            args,
            writer
        )

    if val_loaders is not None:
        val_metrics = validation_step(
            args.train_iters,
            val_loaders,
            model,
            args,
            writer,
            val_iters=-1
        )

    # Last checkpoint
    checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    torch.save(model_dict, checkpoint)
