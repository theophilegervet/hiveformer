import io
import random
import os
from collections import defaultdict

import cv2
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import trange
import wandb
import tap

from utils.utils_without_rlbench import (
    load_instructions,
    count_parameters,
    get_max_episode_length,
    get_gripper_loc_bounds
)
from dataset import RLBenchDataset, RLBenchDatasetWhole
from typing import List, Tuple, Optional
from pathlib import Path


class Arguments(tap.Tap):
    master_port: str = '29500'
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    image_size: str = "256,256"
    max_tries: int = 10
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    checkpoint_freq: int = 10

    # Training and validation datasets
    dataset: List[Path]
    valset: Optional[Tuple[Path, ...]] = None
    dense_interpolation: int = 0
    interpolation_length: int = 100
    trim_to_fixed_len: Optional[int] = None
    train_diffusion_on_whole: int = 0

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    logger: Optional[str] = "tensorboard"  # One of "wandb", "tensorboard", None
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

    # ---------------------------------------------------------------
    # Original HiveFormer parameters
    # ---------------------------------------------------------------

    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1

    # ---------------------------------------------------------------
    # Our non-analogical baseline parameters
    # ---------------------------------------------------------------

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling
    point_cloud_rotate_yaw_range: float = 0.0  # in degrees, 0.0 for no rotation

    visualize_rgb_attn: int = 0  # deactivate by default during training as this has memory overhead
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    single_task_gripper_loc_bounds: int = 0
    gripper_bounds_buffer: float = 0.04

    # Loss
    position_prediction_only: int = 0
    position_loss: str = "ce"  # one of "ce" (our model), "mse" (original HiveFormer)
    ground_truth_gaussian_spread: float = 0.01
    compute_loss_at_all_layers: int = 0
    position_loss_coeff: float = 1.0
    position_offset_loss_coeff: float = 10000.0
    rotation_loss_coeff: float = 10.0
    symmetric_rotation_loss: int = 0
    gripper_loss_coeff: float = 1.0
    label_smoothing: float = 0.0
    regress_position_offset: int = 0

    # Ghost points
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_ghost_points: int = 1000
    num_ghost_points_val: int = 10000
    use_ground_truth_position_for_sampling_train: int = 1  # considerably speeds up training
    use_ground_truth_position_for_sampling_val: int = 0    # for debugging

    # Model
    action_dim: int = 7
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 8
    num_vis_ins_attn_layers: int = 2
    # one of "quat_from_top_ghost", "quat_from_query", "6D_from_top_ghost", "6D_from_query"
    rotation_parametrization: str = "quat_from_query"
    use_instruction: int = 0
    use_goal: int = 0
    use_goal_at_test: int = 1
    use_rgb: int = 1
    task_specific_biases: int = 0
    diffusion_head: str = "simple"

    # Positional features
    positional_features: Optional[str] = "none"  # one of "xyz_concat", "z_concat", "xyz_add", "z_add", "none"

    # ---------------------------------------------------------------
    # Our analogical network additional parameters
    # ---------------------------------------------------------------

    support_set: str = "others"  # one of "self" (for debugging), "others"
    support_set_size: int = 1
    global_correspondence: int = 0
    num_matching_cross_attn_layers: int = 2


def training(
    rank,
    world_size,
    model,
    val_loaders,
    args,
    gripper_loc_bounds
):
    setup(rank, world_size, args.master_port)

    train_loader = get_train_loader(rank, world_size, args, gripper_loc_bounds)

    model = model.to(rank)
    model = DDP(model, [rank], find_unused_parameters=True)
    model_no_ddp = model.module

    # Set up optimizer
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": args.lr},
        {"params": [], "weight_decay": 5e-4, "lr": args.lr},
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in model_no_ddp.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)
    optimizer = optim.AdamW(optimizer_grouped_parameters)

    # Load checkpoint
    start_iter = 0
    best_loss = None
    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model_dict_weight = {}
        for key in model_dict["weight"]:
            _key = key[7:]
            model_dict_weight[_key] = model_dict["weight"][key]
        model_no_ddp.load_state_dict(model_dict_weight)
        optimizer.load_state_dict(model_dict["optimizer"])
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

    model.train()

    # Set up logging and checkpointing
    if rank == 0:
        if args.logger == "tensorboard":
            writer = SummaryWriter(log_dir=args.log_dir)
        elif args.logger == "wandb":
            wandb.init(project="analogical_manipulation")
            wandb.run.name = str(args.log_dir).split("/")[-1]
            wandb.config.update(args.__dict__)
            writer = None
        else:
            writer = None

        aggregated_losses = defaultdict(list)

    iter_loader = iter(train_loader)

    with trange(start_iter, args.train_iters) as tbar:
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

            if step_id % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if rank == 0:
                aggregated_losses["noise_mse"].append(loss)

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
                        # skip train to save time
                        # validation_step(
                        #     step_id,
                        #     [train_loader],
                        #     model,
                        #     args,
                        #     writer,
                        #     val_iters=int(24/args.batch_size),
                        #     split='train'
                        # )
                        val_metrics = validation_step(
                            step_id,
                            val_loaders,
                            model,
                            args,
                            writer,
                            val_iters=int(4*len(args.tasks)/args.batch_size_val)
                        )
                        model.train()
                    else:
                        val_metrics = {}
                    m_ = 'val-loss-0/pos_l2'
                    if m_ not in val_metrics:
                        print(f'{m_} is not reported, storing unconditionally')
                    new_loss = val_metrics.get(m_, None)
                    if new_loss is None or best_loss is None or new_loss <= best_loss:
                        best_loss = new_loss
                        torch.save({
                            "weight": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter": step_id + 1,
                            "best_loss": best_loss
                        }, args.log_dir / "best.pth")
                    torch.save({
                        "weight": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": step_id + 1,
                        "best_loss": best_loss
                    }, args.log_dir / "last.pth")

                    # torch.save({
                    #     "weight": model.state_dict(),
                    #     "optimizer": optimizer.state_dict(),
                    #     "iter": step_id + 1,
                    #     "best_loss": best_loss
                    # }, args.log_dir / f"model.step={step_id+1}.pth")


@torch.no_grad()
def validation_step(
    step_id,
    val_loaders,
    model,
    args,
    writer=None,
    val_iters=10,
    split='val'
):
    values = {}
    device = next(model.parameters()).device
    model.eval()

    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            action = model.module.compute_trajectory(
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"].to(device),
                sample["curr_gripper"].to(device),
                sample["action"].to(device)
            )
            losses, losses_B = compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )
            for n, l in losses.items():
                key = f"{split}-loss-{val_id}/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss-{val_id}/{task}/{n}"
                    l_task = l[tasks==task].mean()
                    # print(l)
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])
                
            # # Generate visualizations
            # if i == 0:
            #     viz_key = f'{split}-viz-{val_id}/viz'
            #     viz = generate_visualizations(
            #         action,
            #         sample["trajectory"].to(device),
            #         sample["trajectory_mask"].to(device)
            #     )
            #     if args.logger == 'tensorboard':
            #         writer.add_image(viz_key, viz, step_id)

        values = {k: v.mean().item() for k, v in values.items()}
        for key, val in values.items():
            if args.logger == "tensorboard":
                writer.add_scalar(key, val, step_id)
            elif args.logger == "wandb":
                wandb.log({key: val}, step=step_id)

        if args.logger == "tensorboard":
            writer.add_scalar(f"lr/", args.lr, step_id)
        elif args.logger == "wandb":
            wandb.log({"lr": args.lr}, step=step_id)

        print(f"Step {step_id}:")
        for key, value in values.items():
            print(f"{key}: {value:.03f}")

    return values


def compute_metrics(pred, gt, mask):
    # pred/gt are (B, L, 7), mask (B, L)
    mask = mask.float()
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt() * (1 - mask)
    quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1) * (1 - mask)
    div_ = (1 - mask).sum()
    div_batch = (1 - mask).sum(-1)

    return {
        'action_mse': (
            F.mse_loss(pred, gt, reduction='none')* (1 - mask)[..., None]
        ).mean(-1).sum() / div_,
        'pos_l2': pos_l2.sum() / div_,
        'pos_acc_001': ((pos_l2 < 0.01).float()  * (1 - mask)).sum() / div_,
        'rot_l1': quat_l1.sum() / div_,
        'rot_l1_005': ((quat_l1 < 0.05).float()  * (1 - mask)).sum() / div_,
        'rot_l1_0025': ((quat_l1 < 0.025).float()  * (1 - mask)).sum() / div_,
    }, {
        'pos_l2': pos_l2.sum(-1) / div_batch,
        'pos_acc_001': ((pos_l2 < 0.01).float()  * (1 - mask)).sum(-1) / div_batch,
        'rot_l1': quat_l1.sum(-1) / div_batch,
        'rot_l1_005': ((quat_l1 < 0.05).float()  * (1 - mask)).sum(-1) / div_batch,
        'rot_l1_0025': ((quat_l1 < 0.025).float()  * (1 - mask)).sum(-1) / div_batch,
    }


def generate_visualizations(pred, gt, mask, box_size=0.3):
    batch_idx = 0
    pred = pred[batch_idx].detach().cpu().numpy()
    gt = gt[batch_idx].detach().cpu().numpy()
    mask = mask[batch_idx].detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2], 
        color='red', label='pred'
    )
    ax.scatter3D(
        gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2], 
        color='blue', label='gt'
    )

    center = gt[~mask].mean(0)
    ax.set_xlim(center[0] - box_size, center[0] + box_size)
    ax.set_ylim(center[1] - box_size, center[1] + box_size)
    ax.set_zlim(center[2] - box_size, center[2] + box_size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.legend()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    img = fig_to_numpy(fig, dpi=120)
    plt.close()
    return img.transpose(2, 0, 1)


def fig_to_numpy(fig, dpi=60):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def collate_fn(batch):
    # Dynamic padding for trajectory
    max_len = max(batch[b]['trajectory_len'].max() for b in range(len(batch)))
    for item in batch:
        h, n, c = item['trajectory'].shape
        traj = torch.zeros(h, max_len, c)
        traj[:, :n] = item['trajectory']
        item['trajectory'] = traj

    # Unfold multi-step demos to form a longer batch
    keys = [
        "trajectory", "trajectory_len",
        "rgbs", "pcds", "curr_gripper", "action", "instr"
    ]
    ret_dict = {key: torch.cat([item[key] for item in batch]) for key in keys}

    # Trajectory mask
    trajectory_mask = torch.zeros(ret_dict['trajectory'].shape[:-1])
    for i, len_ in enumerate(ret_dict['trajectory_len']):
        trajectory_mask[i, len_:] = 1
    ret_dict["trajectory_mask"] = trajectory_mask.bool()

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict


def get_train_loader(rank, world_size, args, gripper_loc_bounds):
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

    DSet = RLBenchDatasetWhole if args.train_diffusion_on_whole else RLBenchDataset
    dataset = DSet(
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
        interpolation_length=args.interpolation_length,
        dense_interpolation=bool(args.dense_interpolation),
        return_low_lvl_trajectory=True,
        action_dim=args.action_dim,
        trim_to_fixed_len=args.trim_to_fixed_len
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        sampler=sampler
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
        DSet = RLBenchDatasetWhole if args.train_diffusion_on_whole else RLBenchDataset
        dataset = DSet(
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
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            action_dim=args.action_dim,
            trim_to_fixed_len=args.trim_to_fixed_len
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size_val,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        loaders.append(loader)

    return loaders


def get_model(args, gripper_loc_bounds):
    if args.model == "diffusion":
        from model.diffusion_planner.diffusion_model import DiffusionPlanner
        _model = DiffusionPlanner(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            output_dim=args.action_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            num_sampling_level=args.num_sampling_level,
            use_instruction=bool(args.use_instruction),
            use_goal=bool(args.use_goal),
            use_goal_at_test=bool(args.use_goal_at_test),
            use_rgb=bool(args.use_rgb),
            gripper_loc_bounds=gripper_loc_bounds,
            positional_features=args.positional_features,
            diffusion_head=args.diffusion_head
        )
    elif args.model == "regression":
        from model.trajectory_regressor.trajectory_model import TrajectoryRegressor
        _model = TrajectoryRegressor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            output_dim=args.action_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            num_sampling_level=args.num_sampling_level,
            use_instruction=bool(args.use_instruction),
            use_goal=bool(args.use_goal),
            use_rgb=bool(args.use_rgb),
            gripper_loc_bounds=gripper_loc_bounds,
            positional_features=args.positional_features
        )
    elif args.model == "continuous_diffusion":
        from model.diffusion_planner.diffusion_continuous import DiffusionPlanner
        _model = DiffusionPlanner(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            output_dim=args.action_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            num_sampling_level=args.num_sampling_level,
            use_instruction=bool(args.use_instruction),
            use_goal=bool(args.use_goal),
            use_goal_at_test=bool(args.use_goal_at_test),
            use_rgb=bool(args.use_rgb),
            gripper_loc_bounds=gripper_loc_bounds,
            positional_features=args.positional_features,
            diffusion_head=args.diffusion_head
        )

    model_params = count_parameters(_model)
    print("Model parameters:", model_params)

    return _model


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_log_dir(args):
    return args.base_log_dir / args.exp_log_dir / args.run_log_dir


if __name__ == "__main__":
    args = Arguments().parse_args()

    print()
    print("Arguments:")
    print(args)

    print()
    print("-" * 100)
    print()

    log_dir = get_log_dir(args)
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(log_dir / "hparams.json"))

    print("Logging:", log_dir)
    print("Args n_gpus:", args.n_gpus)
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

    model = get_model(args, gripper_loc_bounds)

    print()
    print("-" * 100)
    print()

    val_loaders = get_val_loaders(args, gripper_loc_bounds)

    if args.train_iters > 0:
        # DDP training
        world_size = args.n_gpus
        training_args = (
            world_size,
            model,
            val_loaders,
            args,
            gripper_loc_bounds
        )
        mp.spawn(training, args=training_args, nprocs=world_size, join=True)
