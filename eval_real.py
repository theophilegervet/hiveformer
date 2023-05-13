import sys
import random
from typing import Tuple, Optional
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import tap
import json
from filelock import FileLock
from scipy.spatial.transform import Rotation as R

from train import Arguments as TrainArguments
from model.released_hiveformer.network import Hiveformer
from model.non_analogical_baseline.baseline import Baseline
from model.analogical_network.analogical_network import AnalogicalNetwork
from utils.utils_with_rlbench import (
    RLBenchEnv,
    Actioner,
)
from utils.utils_without_rlbench import (
    load_episodes,
    load_instructions,
    get_max_episode_length,
    get_gripper_loc_bounds,
)

sys.path.append('/home/zhouxian/git/franka')
from frankapy import FrankaArm
from utils.utils_without_rlbench import TASK_TO_ID
from camera.kinect import Kinect

class Arguments(tap.Tap):
    checkpoint: Path
    seed: int = 2
    save_img: bool = True
    device: str = "cuda"
    num_episodes: int = 1
    headless: bool = False
    max_tries: int = 10
    # max_tries: int = 30
    output: Path = Path(__file__).parent / "records.txt"
    record_actions: bool = False
    replay_actions: Optional[Path] = None
    ground_truth_rotation: bool = False
    ground_truth_position: bool = False
    ground_truth_gripper: bool = False
    task = None
    instructions: Optional[Path] = "instructions.pkl"
    arch: Optional[str] = None
    variation: int = 0
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    
    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "eval_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    
    # Toggle to switch between offline and online evaluation
    offline: int = 0

    # Toggle to switch between original HiveFormer and our models
    model: str = "baseline"  # one of "original", "baseline", "analogical"

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
    gripper_loc_bounds_file: str = "tasks/10_autolambda_tasks_location_bounds.json"
    single_task_gripper_loc_bounds: int = 0
    gripper_bounds_buffer: float = 0.01

    position_prediction_only: int = 0
    regress_position_offset: int = 0
    max_episodes: int = 0

    # Ghost points
    num_sampling_level: int = 3
    coarse_to_fine_sampling: int = 1
    fine_sampling_ball_diameter: float = 0.16
    weight_tying: int = 1
    gp_emb_tying: int = 0
    simplify: int = 0
    simplify_ins: int = 0
    ins_pos_emb: int = 0
    vis_ins_att: int = 0
    vis_ins_att_complex: int = 0
    num_ghost_points: int = 1000
    num_ghost_points_val: int = 1000
    disc_rot: int = 0
    disc_rot_res: float = 5.0

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 60
    num_ghost_point_cross_attn_layers: int = 2
    num_query_cross_attn_layers: int = 2
    separate_coarse_and_fine_layers: int = 1
    rotation_parametrization: str = "quat_from_query"  # one of "quat_from_top_ghost", "quat_from_query" for now
    use_instruction: int = 0

    # ---------------------------------------------------------------
    # Our analogical network additional parameters
    # ---------------------------------------------------------------

    support_set: str = "others"  # one of "self" (for debugging), "others"
    support_set_size: int = 1
    global_correspondence: int = 0
    num_matching_cross_attn_layers: int = 2
    task_specific_parameters: int = 0
    randomize_vp: int = 0


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.base_log_dir / args.exp_log_dir

    def get_log_file(version):
        log_file = f"{args.run_log_dir}_version{version}"
        return log_file

    version = 0
    while (log_dir / get_log_file(version)).is_dir():
        version += 1

    return log_dir / get_log_file(version)


def copy_args(checkpoint: Path, args: Arguments) -> Arguments:
    args = deepcopy(args)

    print("Copying args from", checkpoint)

    # Update args accordingly:
    hparams = checkpoint.parent / "hparams.json"
    print(hparams, hparams.is_file())
    if hparams.is_file():
        print("Loading args from checkpoint")
        train_args = TrainArguments()
        train_args.load(str(hparams))
        for key in args.class_variables:
            v = getattr(args, key)
            if v is None and key in train_args.class_variables:
                setattr(args, key, getattr(train_args, key))
                print("Copying", key, ":", getattr(args, key))

    return args


def load_model(checkpoint: Path, args: Arguments) -> Hiveformer:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    print("Loading model from", checkpoint, flush=True)

    if (
        args.depth is None
        or args.dim_feedforward is None
        or args.hidden_dim is None
        or args.instr_size is None
        or args.mask_obs_prob is None
        or args.num_layers is None
    ):
        raise ValueError("Please provide the missing parameters")

    task = args.task
    variation = args.variation
    max_episode_length = get_max_episode_length((task,), (variation,))

    # Gripper workspace is the union of workspaces for all tasks
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file, task=task, buffer=args.gripper_bounds_buffer)

    model = Baseline(
        backbone=args.backbone,
        image_size=tuple(int(x) for x in args.image_size.split(",")),
        embedding_dim=args.embedding_dim,
        num_ghost_point_cross_attn_layers=args.num_ghost_point_cross_attn_layers,
        num_query_cross_attn_layers=args.num_query_cross_attn_layers,
        rotation_parametrization=args.rotation_parametrization,
        gripper_loc_bounds=gripper_loc_bounds,
        num_ghost_points=args.num_ghost_points,
        num_ghost_points_val=args.num_ghost_points_val,
        weight_tying=bool(args.weight_tying),
        gp_emb_tying=bool(args.gp_emb_tying),
        simplify=bool(args.simplify),
        simplify_ins=bool(args.simplify_ins),
        ins_pos_emb=bool(args.ins_pos_emb),
        vis_ins_att=bool(args.vis_ins_att),
        vis_ins_att_complex=bool(args.vis_ins_att_complex),
        disc_rot=bool(args.disc_rot),
        disc_rot_res=args.disc_rot_res,
        num_sampling_level=args.num_sampling_level,
        fine_sampling_ball_diameter=args.fine_sampling_ball_diameter,
        regress_position_offset=bool(args.regress_position_offset),
        visualize_rgb_attn=bool(args.visualize_rgb_attn),
        use_instruction=bool(args.use_instruction),
    ).to(device)


    model_dict = torch.load(checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        # if 'prediction_head.feature_pyramid.inner_blocks' in _key:
        #     _key = _key[:46] + _key[48:]
        # if 'prediction_head.feature_pyramid.layer_blocks' in _key:
        #     _key = _key[:46] + _key[48:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)

    model.eval()

    return model


def find_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_dir():
        candidates = [c for c in checkpoint.rglob("*.pth") if c.name != "best"]
        candidates = sorted(candidates, key=lambda p: p.lstat().st_mtime)
        assert candidates != [], checkpoint
        return candidates[-1]

    return checkpoint

def transform(rgb, pc):
    # normalise to [-1, 1]
    rgb = 2 * (rgb / 255.0 - 0.5)

    if rgb.shape == pc.shape == (720, 1080, 3):
        rgb = rgb[::3, ::3]
        pc = pc[::3, ::3]
    else:
        assert False

    return rgb, pc

if __name__ == "__main__":
    args = Arguments().parse_args()

    fa = FrankaArm()
    kinect = Kinect()

    print('Reset...')
    fa.reset_joints()
    print('Open gripper...')
    fa.open_gripper()
    gripper_open = True

    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("log dir", log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    checkpoint = find_checkpoint(args.checkpoint)
    args = copy_args(checkpoint, args)
    if checkpoint is None:
        raise RuntimeError()
    model = load_model(checkpoint, args)
    device = args.device




    # Evaluate
    instructions = load_instructions(args.instructions)
    if instructions is None:
        raise NotImplementedError()
    max_eps_dict = load_episodes()["max_episode_length"]

    max_episodes = max_eps_dict[args.task] if args.max_episodes == 0 else args.max_episodes
    instr = random.choice(instructions[args.task][args.variation]).unsqueeze(0).to(device)
    task_id = torch.tensor(TASK_TO_ID[args.task]).unsqueeze(0).to(device)
    
    for step_id in range(max_episodes):
        print(step_id)
        # get obs
        rgb = kinect.get_rgb()[:, 100:-100, :]
        pcd = kinect.get_pc()[:, 100:-100, :]
        rgb, pcd = transform(rgb, pcd)
        rgb = rgb.transpose((2, 0, 1))
        pcd = pcd.transpose((2, 0, 1))
        rgbs = torch.tensor(rgb).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        pcds = torch.tensor(pcd).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()

        gripper_pose = fa.get_pose()
        gripper_trans = gripper_pose.translation
        gripper_quat = R.from_matrix(gripper_pose.rotation).as_quat()
        gripper = np.concatenate([gripper_trans, gripper_quat, [gripper_open]])
        gripper = torch.tensor(gripper).to(device).unsqueeze(0).unsqueeze(0).float()

        padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()

        pred = model(
            rgbs,
            pcds,
            padding_mask,
            instr,
            gripper,
            task_id,
        )
        action = model.compute_action(pred).detach().cpu().numpy()  # type: ignore
        action_gripper_open = action[0, -1] > 0.5

        # move
        target_pose = fa.get_pose()
        target_pose.translation = action[0, :3]
        target_pose.rotation = R.from_quat(action[0, 3:7]).as_matrix()

        fa.goto_pose(target_pose, duration=5)
        if gripper_open and not action_gripper_open:
            fa.close_gripper()
            gripper_open = False
        elif not gripper_open and action_gripper_open:
            fa.open_gripper()
            gripper_open = True




