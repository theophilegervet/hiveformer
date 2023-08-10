import random
from typing import Tuple, Optional
from pathlib import Path
import torch
import numpy as np
import tap
import json
import os

from model.keypose_optimization.act3d import Baseline
from model.trajectory_optimization.diffusion_model import DiffusionPlanner
from model.trajectory_optimization.regression_model import TrajectoryRegressor
from utils.utils_with_rlbench import (
    RLBenchEnv,
    Actioner,
)
from utils.utils_without_rlbench import (
    load_episodes,
    load_instructions,
    get_max_episode_length,
    get_gripper_loc_bounds,
    TASK_TO_ID,
    round_floats
)


class Arguments(tap.Tap):
    checkpoint: Path
    act3d_checkpoint: Path
    seed: int = 2
    save_img: bool = True
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    record_actions: bool = False
    replay_actions: Optional[Path] = None
    ground_truth_rotation: bool = False
    ground_truth_position: bool = False
    ground_truth_gripper: bool = False
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    arch: Optional[str] = None
    variations: Tuple[int, ...] = (0,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    
    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    
    # Toggle to switch between offline and online evaluation
    # 0: false, 1: keypose 2: full
    offline: int = 0

    # Toggle to switch between original HiveFormer and our models
    model: str = "baseline"  # one of "original", "baseline", "analogical"
    traj_model: str = "diffusion"  # one of "original", "baseline", "analogical"

    record_videos: int = 0
    max_steps: int = 50
    collision_checking: int = 0
    use_rgb: int = 1
    use_goal: int = 0
    use_goal_at_test: int = 1
    dense_interpolation: int = 0
    interpolation_length: int = 100

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

    visualize_rgb_attn: int = 0
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    act3d_gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    single_task_gripper_loc_bounds: int = 0
    gripper_bounds_buffer: float = 0.04
    act3d_gripper_bounds_buffer: float = 0.04

    position_prediction_only: int = 0
    regress_position_offset: int = 0

    # Ghost points
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000

    # Model
    action_dim: int = 7
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    act3d_num_query_cross_attn_layers: int = 2
    num_vis_ins_attn_layers: int = 2
    # one of "quat_from_top_ghost", "quat_from_query", "6D_from_top_ghost", "6D_from_query"
    rotation_parametrization: str = "quat_from_query"
    use_instruction: int = 0
    act3d_use_instruction: int = 0
    task_specific_biases: int = 0

    # Positional features
    positional_features: Optional[str] = "none"  # one of "xyz_concat", "z_concat", "xyz_add", "z_add", "none"

    # ---------------------------------------------------------------
    # Our analogical network additional parameters
    # ---------------------------------------------------------------

    support_set: str = "others"  # one of "self" (for debugging), "others"
    support_set_size: int = 1
    global_correspondence: int = 0
    num_matching_cross_attn_layers: int = 2


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)
    print("Loading model from", args.act3d_checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    diffusion_gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_bounds_buffer
    )
    act3d_gripper_loc_bounds = get_gripper_loc_bounds(
        args.act3d_gripper_loc_bounds_file,
        task=task, buffer=args.act3d_gripper_bounds_buffer
    )

    if args.traj_model == "diffusion":
        diffusion_model = DiffusionPlanner(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            output_dim=args.action_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            num_sampling_level=args.num_sampling_level,
            use_instruction=bool(args.use_instruction),
            num_query_cross_attn_layers=args.num_query_cross_attn_layers,
            use_goal=bool(args.use_goal),
            use_goal_at_test=bool(args.use_goal_at_test),
            gripper_loc_bounds=diffusion_gripper_loc_bounds,
            positional_features=args.positional_features
        ).to(device)
    elif args.traj_model == "regression":
        diffusion_model = TrajectoryRegressor(
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
            gripper_loc_bounds=diffusion_gripper_loc_bounds,
            positional_features=args.positional_features
        ).to(device)
    act3d_model = Baseline(
        backbone=args.backbone,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        embedding_dim=args.embedding_dim,
        num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
        num_query_cross_attn_layers=args.act3d_num_query_cross_attn_layers,
        rotation_parametrization=args.rotation_parametrization,
        gripper_loc_bounds=act3d_gripper_loc_bounds,
        num_ghost_points=args.num_ghost_points,
        num_ghost_points_val=args.num_ghost_points_val,
        weight_tying=bool(args.weight_tying),
        gp_emb_tying=bool(args.gp_emb_tying),
        num_sampling_level=args.num_sampling_level,
        fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
        regress_position_offset=bool(args.regress_position_offset),
        visualize_rgb_attn=bool(args.visualize_rgb_attn),
        use_instruction=bool(args.act3d_use_instruction),
        task_specific_biases=bool(args.task_specific_biases),
        positional_features=args.positional_features,
        task_ids=[TASK_TO_ID[task] for task in args.tasks],
    ).to(device)

    diffusion_model_dict = torch.load(args.checkpoint, map_location="cpu")
    diffusion_model_dict_weight = {}
    for key in diffusion_model_dict["weight"]:
        _key = key[7:]
        diffusion_model_dict_weight[_key] = diffusion_model_dict["weight"][key]
    diffusion_model.load_state_dict(diffusion_model_dict_weight)
    diffusion_model.eval()

    act3d_model_dict = torch.load(args.act3d_checkpoint, map_location="cpu")
    act3d_model_dict_weight = {}
    for key in act3d_model_dict["weight"]:
        _key = key[7:]
        act3d_model_dict_weight[_key] = act3d_model_dict["weight"][key]
    act3d_model.load_state_dict(act3d_model_dict_weight)
    act3d_model.eval()

    return diffusion_model, act3d_model


if __name__ == "__main__":
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
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    traj_model, keypose_model = load_models(args)

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        traj_cmd=args.predict_keypose,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking)
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(
        keypose_model=keypose_model,
        traj_model=traj_model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_keypose=args.predict_keypose,
        predict_trajectory=args.predict_trajectory
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=(
                max_eps_dict[task_str] if args.max_steps == -1
                else args.max_steps
            ),
            num_variations=args.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            log_dir=log_dir / task_str if args.save_img else None,
            max_tries=args.max_tries,
            save_attn=False,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            record_videos=bool(args.record_videos),
            position_prediction_only=bool(args.position_prediction_only),
            offline=args.offline,
            verbose=bool(args.verbose),
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
