from model.utils.utils import normalise_quat
from scipy.interpolate import CubicSpline, interp1d
import itertools
import random
import blosc
import pickle
from typing import (
    Union,
    Optional,
    Tuple,
    List,
    Dict,
    Callable,
    TypeVar,
    Generic,
)
from pickle import UnpicklingError
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data._utils.collate import default_collate
import einops
try:
    from pytorch3d import transforms as torch3d_tf
except:
    pass

from utils.utils_without_rlbench import (
    Instructions, Sample, Camera, TASK_TO_ID, load_episodes
)


T = TypeVar("T")
U = TypeVar("U")


class Cache(Generic[T, U]):
    def __init__(self, size: int, loader: Callable[[T], U]):
        self._size = size
        self._loader = loader
        self._keys: List[T] = []
        self._cache: Dict[T, U] = {}

    def __call__(self, args: T) -> U:
        if self._size == 0:
            return self._loader(args)

        if args in self._cache:
            index = self._keys.index(args)
            del self._keys[index]
            self._keys.append(args)
            return self._cache[args]

        # print(args, len(self._keys), self._size)
        value = self._loader(args)

        if len(self._keys) == self._size and self._keys != []:
            key = self._keys[0]
            del self._cache[key]
            del self._keys[0]

        if len(self._keys) < self._size:
            self._keys.append(args)
            self._cache[args] = value

        return value


def data_transform(scales, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Expect tensors as T, N, C, H, W
    """
    keys = list(kwargs.keys())

    if len(keys) == 0:
        raise RuntimeError("No args")

    # Continuous range of scales
    sc = np.random.uniform(*scales)

    t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
    kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
    resized_size = [int(raw_h * sc), int(raw_w * sc)]

    # Resize based on randomly sampled scale
    kwargs = {
        n: transforms_f.resize(
            arg,
            resized_size,
            transforms.InterpolationMode.NEAREST
            # if "pc" in n
            # else transforms.InterpolationMode.BILINEAR,
        )
        for n, arg in kwargs.items()
    }

    # Adding padding if crop size is smaller than the resized size
    if raw_h > resized_size[0] or raw_w > resized_size[1]:
        right_pad, bottom_pad = max(raw_h - resized_size[1], 0), max(
            raw_w - resized_size[0], 0
        )
        kwargs = {
            n: transforms_f.pad(
                arg,
                padding=[0, 0, right_pad, bottom_pad],
                padding_mode="reflect",
            )
            for n, arg in kwargs.items()
        }

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(
        kwargs[keys[0]], output_size=(raw_h, raw_w)
    )

    kwargs = {n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()}

    kwargs = {
        n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
        for n, arg in kwargs.items()
    }

    return kwargs


def loader(file: Path) -> Optional[np.ndarray]:
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


class Resize:
    """
    Resize and pad/crop the image and aligned point cloud.
    """
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Accept tensors as T, N, C, H, W
        """
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_h - resized_size[1], 0), max(
                raw_w - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()}

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class Rotate:
    """
    Rotate the point cloud, current gripper, and next ground-truth gripper, while
    ensuring the current gripper and next ground-truth gripper stay within workspace
    bounds.
    """
    def __init__(self, gripper_loc_bounds, yaw_range, num_tries=10):
        self.gripper_loc_bounds = torch.from_numpy(gripper_loc_bounds)
        self.yaw_range = np.deg2rad(yaw_range)
        self.num_tries = num_tries

    def __call__(self, pcds, gripper, action, mask, trajectory=None):
        if self.yaw_range == 0.0:
            return pcds, gripper, action, trajectory

        augmentation_rot_4x4 = self._sample_rotation()
        gripper_rot_4x4 = self._gripper_action_to_matrix(gripper)
        action_rot_4x4 = self._gripper_action_to_matrix(action)
        if trajectory is not None:
            trajectory = einops.rearrange(trajectory, 'b l c -> (b l ) c')
            traj_rot = self._gripper_action_to_matrix(trajectory)

        for i in range(self.num_tries):
            gripper_rot_4x4 = augmentation_rot_4x4 @ gripper_rot_4x4
            action_rot_4x4 = augmentation_rot_4x4 @ action_rot_4x4

            gripper_position, gripper_quaternion = self._gripper_matrix_to_action(gripper_rot_4x4)
            action_position, action_quaternion = self._gripper_matrix_to_action(action_rot_4x4)
            if trajectory is not None:
                traj_position, traj_quaternion = self._gripper_matrix_to_action(traj_rot)

            if self._check_bounds(gripper_position[mask], action_position[mask]):
                gripper[mask, :3], gripper[mask, 3:7] = gripper_position[mask], gripper_quaternion[mask]
                action[mask, :3], action[mask, 3:7] = action_position[mask], action_quaternion[mask]
                if trajectory is not None:
                    trajectory[:, :3], trajectory[:, 3:7] = traj_position, traj_quaternion
                    trajectory = trajectory.reshape(len(action), -1, trajectory.size(-1))
                pcds[mask] = einops.einsum(
                    augmentation_rot_4x4[:3, :3], pcds[mask], "c2 c1, t ncam c1 h w -> t ncam c2 h w")
                break

        return pcds, gripper, action, trajectory

    def _check_bounds(self, gripper_position, action_position):
        return (
            (gripper_position >= self.gripper_loc_bounds[0]).all() and
            (gripper_position <= self.gripper_loc_bounds[1]).all() and
            (action_position >= self.gripper_loc_bounds[0]).all() and
            (action_position <= self.gripper_loc_bounds[1]).all()
        )

    def _sample_rotation(self):
        yaw = 2 * self.yaw_range * torch.rand(1) - self.yaw_range
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        rot_3x3 = torch3d_tf.euler_angles_to_matrix(torch.stack([roll, pitch, yaw], dim=1), "XYZ")
        rot_4x4 = torch.eye(4)
        rot_4x4[:3, :3] = rot_3x3
        return rot_4x4

    def _gripper_action_to_matrix(self, action):
        position = action[:, :3]
        quaternion = action[:, [6, 3, 4, 5]]
        rot_3x3 = torch3d_tf.quaternion_to_matrix(quaternion)
        rot_4x4 = torch.eye(4).unsqueeze(0).repeat(position.shape[0], 1, 1)
        rot_4x4[:, :3, :3] = rot_3x3
        rot_4x4[:, :3, 3] = position
        return rot_4x4

    def _gripper_matrix_to_action(self, matrix):
        position = matrix[:, :3, 3]
        rot_3x3 = matrix[:, :3, :3]
        quaternion = torch3d_tf.matrix_to_quaternion(rot_3x3)[:, [1, 2, 3, 0]]
        return position, quaternion


class RLBenchDataset(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        root: Union[Path, str, List[Path], List[str]],
        image_size: Tuple[int, int],
        taskvar: List[Tuple[str, int]],
        instructions: Instructions,
        max_episode_length: int,
        cache_size: int,
        max_episodes_per_task: int,
        gripper_loc_bounds=None,
        num_iters: Optional[int] = None,
        cameras: Tuple[Camera, ...] = ("wrist", "left_shoulder", "right_shoulder"),
        training: bool = True,
        image_rescale=(1.0, 1.0),
        point_cloud_rotate_yaw_range=0.0,
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        action_dim=8,
        trim_to_fixed_len=None,
        train_diffusion_on_whole=False
    ):
        self._cache = Cache(cache_size, loader)
        self._cameras = cameras
        self._image_size = image_size
        self._max_episode_length = max_episode_length
        self._max_episodes_per_task = max_episodes_per_task
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        self._action_dim = action_dim
        self._trim_to_fixed_len = trim_to_fixed_len
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root: List[Path] = [Path(r).expanduser() for r in root]
        self._dense_interpolation = dense_interpolation
        self._interpolation_length = interpolation_length
        self._train_diffusion_on_whole = train_diffusion_on_whole

        max_episode_length_dict = load_episodes()["max_episode_length"]

        # We keep only useful instructions to save mem
        self._instructions: Instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1
            else:
                print(f"Can't find dataset folder {data_dir}")

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)
            self._rotate = Rotate(
                gripper_loc_bounds=gripper_loc_bounds,
                yaw_range=point_cloud_rotate_yaw_range
            )

        # File-names of episodes per-task and variation
        self._data_dirs = []
        episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]  # Backward compatibility
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            episodes = npy_episodes + dat_episodes
            # Split episodes equally into task variations
            episodes = episodes[:self._max_episodes_per_task // self._num_vars[task] + 1]
            num_episodes = len(episodes)
            if num_episodes == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            self._data_dirs.append(data_dir)
            episodes_by_task[task] += episodes

        # All episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > self._max_episodes_per_task:
                eps = random.sample(eps, self._max_episodes_per_task)
            history_truncated_eps = []
            # Chunk too long episodes (repeat episode with diff chunk id)
            for (task, var, ep) in eps:
                chunks = math.ceil(max_episode_length_dict[task] / self._max_episode_length)
                for chunk in range(chunks):
                    history_truncated_eps.append((task, var, ep, chunk))
            self._episodes += history_truncated_eps
            self._num_episodes += len(history_truncated_eps)

        print(f"Created dataset from {root} with {self._num_episodes} episodes (after chunking "
              f"them by max episode length)")

    def resample_trajectory(self, trajectory):
        trajectory = trajectory.numpy()
        # Calculate the current number of steps
        old_num_steps = len(trajectory)

        # Create a 1D array for the old and new steps
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        # Interpolate each dimension separately
        resampled_trajectory = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == 7: # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])

            resampled_trajectory[:, i] = interpolator(new_steps)

        resampled_trajectory = torch.tensor(resampled_trajectory)
        resampled_trajectory[:, 3:7] = normalise_quat(resampled_trajectory[:, 3:7])
        return resampled_trajectory

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors]  # wrt frame_ids, (1, 8)
            [trajectories],  # wrt frame_ids, (N_i, 8)
        ]
        """
        if self._train_diffusion_on_whole:
            return self._getitem_whole(episode_id)

        episode_id %= self._num_episodes
        task, variation, file, chunk = self._episodes[episode_id]

        # Load episode
        episode = self._cache(file)
        if episode is None:
            return None

        # Get frame ids for this chunk
        frame_ids = episode[0][chunk * self._max_episode_length: (chunk + 1) * self._max_episode_length]
        if len(frame_ids) == 0:
            # Episode ID is not valid, sample another one
            episode_id = random.randint(0, self._num_episodes - 1)
            return self.__getitem__(episode_id)

        # Get the image tensors for the frame ids we got
        states = torch.stack([torch.from_numpy(episode[1][i]) for i in frame_ids])
        if states.shape[-1] != self._image_size[1] or states.shape[-2] != self._image_size[0]:
            raise ValueError(f"{states.shape} {self._episodes[episode_id]}")

        # Camera ids
        cameras = list(episode[3][0].keys())
        assert all(c in cameras for c in self._cameras)
        index = torch.tensor([cameras.index(c) for c in self._cameras])

        # Re-map states based on camera ids
        states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        instr = random.choice(self._instructions[task][variation])
        instr = instr[None].repeat(len(rgbs), 1, 1)

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        if self._dense_interpolation:
            for i in range(len(episode[5])):
                episode[5][i] = self.resample_trajectory(episode[5][i])

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            traj_items = [episode[5][i] for i in frame_ids]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, 8)
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            # Trim to fixed length
            if self._trim_to_fixed_len is not None:
                traj = traj[:, :self._trim_to_fixed_len]  # (n_frames, T, 8)
                traj_lens = traj_lens * 0 + self._trim_to_fixed_len

        # Augmentations
        if self._training:
            pcds, gripper, action, traj = self._rotate(pcds, gripper, action, None, traj)
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        return {
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action[..., :self._action_dim],  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper[..., :self._action_dim],  # e.g. tensor (n_frames, 8), current pose
            "trajectory": traj[..., :self._action_dim],  # e.g. tensor (n_frames, 67, 8)
            "trajectory_len": traj_lens  # e.g. tensor (n_frames,)
        }

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes

    def _getitem_whole(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors]  # wrt frame_ids, (1, 8)
            [trajectories],  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file, chunk = self._episodes[episode_id]

        # Load episode
        episode = self._cache(file)
        if episode is None:
            return None

        # CHeck for valid episode
        frame_ids = episode[0][chunk * self._max_episode_length: (chunk + 1) * self._max_episode_length]
        if len(frame_ids) == 0:
            # Episode ID is not valid, sample another one
            episode_id = random.randint(0, self._num_episodes - 1)
            return self.__getitem__(episode_id)

        # Get the image tensors for the frame ids we got
        states = torch.from_numpy(episode[1][0])[None]

        # Camera ids
        cameras = list(episode[3][0].keys())
        assert all(c in cameras for c in self._cameras)
        index = torch.tensor([cameras.index(c) for c in self._cameras])

        # Re-map states based on camera ids
        states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]  # (1, n_cam, 3, 256, 256)
        pcds = states[:, :, 1]  # (1, n_cam, 3, 256, 256)

        # Get action tensors for respective frame ids
        action = episode[2][-1]

        # Sample one instruction feature
        instr = random.choice(self._instructions[task][variation])
        instr = instr[None].repeat(len(rgbs), 1, 1)

        # Get gripper tensors for respective frame ids
        gripper = episode[4][0]

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            traj = torch.cat(episode[5])
            if self._dense_interpolation:
                traj = self.resample_trajectory(traj)
            traj = traj[None]
            traj_lens = torch.as_tensor([traj.size(1)])
            # Trim to fixed length
            if self._trim_to_fixed_len is not None:
                traj = traj[:, :self._trim_to_fixed_len]  # (n_frames, T, 8)
                traj_lens = traj_lens * 0 + self._trim_to_fixed_len

        # Augmentations
        if self._training:
            pcds, gripper, action, traj = self._rotate(pcds, gripper, action, None, traj)
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        return {
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action[..., :self._action_dim],  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper[..., :self._action_dim],  # e.g. tensor (n_frames, 8), current pose
            "trajectory": traj[..., :self._action_dim],  # e.g. tensor (n_frames, 67, 8)
            "trajectory_len": traj_lens  # e.g. tensor (n_frames,)
        }


class RLBenchAnalogicalDataset(Dataset):
    """
    RLBench analogical dataset:
    - Instead of a single demo, each dataset element is a set of demos from the same task
       where each demo can act as a support set for others
    - During training, all demos in the set come from the same train split, and we use
       each demo in the set for training with all other demos as the support set
    - During evaluation, only one demo in the set comes from the val split while others
       come from the train split and act as the support set
    """

    def __init__(
        self,
        main_root: Union[Path, str],
        support_root: Union[Path, str],
        support_set_size: int,
        image_size: Tuple[int, int],
        taskvar: List[Tuple[str, int]],
        instructions: Instructions,
        max_episode_length: int,
        cache_size: int,
        max_episodes_per_task: int,
        gripper_loc_bounds=None,
        num_iters: Optional[int] = None,
        cameras: Tuple[Camera, ...] = ("wrist", "left_shoulder", "right_shoulder"),
        training: bool = True,
        image_rescale=(1.0, 1.0),
        point_cloud_rotate_yaw_range=0.0,
    ):
        """
        Arguments:
            main_root: path to the main dataset (train split for training, val split for evaluation)
            support_root: path to the support dataset (train split for both training and evaluation)
            support_set_size: number of support episodes for each main episode
        """
        self._cache = Cache(cache_size, loader)
        self._cameras = cameras
        self._image_size = image_size
        self._max_episode_length = max_episode_length
        self._max_episodes_per_task = max_episodes_per_task
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        if isinstance(main_root, (Path, str)):
            main_root = [Path(main_root)]
        if isinstance(support_root, (Path, str)):
            support_root = [Path(support_root)]
        self._main_root = [Path(r).expanduser() for r in main_root]
        self._support_root = [Path(r).expanduser() for r in support_root]
        self._support_set_size = support_set_size
        max_episode_length_dict = load_episodes()["max_episode_length"]

        # We keep only useful instructions to save mem
        self._instructions: Instructions = defaultdict(dict)
        self._num_vars = Counter()
        for root, (task, var) in itertools.product(self._main_root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1
            else:
                print(f"Can't find dataset folder {data_dir}")

        self._resize = Resize(scales=image_rescale)
        self._rotate = Rotate(
            gripper_loc_bounds=gripper_loc_bounds,
            yaw_range=point_cloud_rotate_yaw_range
        )

        # -------------------------------
        # Main episodes
        # -------------------------------

        self._main_data_dirs = []
        main_episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._main_root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                raise ValueError(f"Can't find dataset folder {data_dir}")
            episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            episodes = episodes[: self._max_episodes_per_task // self._num_vars[task] + 1]
            num_episodes = len(episodes)
            if num_episodes == 0:
                raise ValueError(f"Can't find episodes at folder {data_dir}")
            self._main_data_dirs.append(data_dir)
            main_episodes_by_task[task] += episodes

        self._main_episodes = []
        self._main_num_episodes = 0
        for task, eps in main_episodes_by_task.items():
            if len(eps) > self._max_episodes_per_task:
                eps = random.sample(eps, self._max_episodes_per_task)
            history_truncated_eps = []
            for (task, var, ep) in eps:
                chunks = math.ceil(max_episode_length_dict[task] / self._max_episode_length)
                for chunk in range(chunks):
                    history_truncated_eps.append((task, var, ep, chunk))
            self._main_episodes += history_truncated_eps
            self._main_num_episodes += len(history_truncated_eps)

        # -------------------------------
        # Support episodes
        # -------------------------------

        self._support_data_dirs = []
        support_episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._support_root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                raise ValueError(f"Can't find dataset folder {data_dir}")
            episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            episodes = episodes[: self._max_episodes_per_task // self._num_vars[task] + 1]
            num_episodes = len(episodes)
            if num_episodes == 0:
                raise ValueError(f"Can't find episodes at folder {data_dir}")
            self._support_data_dirs.append(data_dir)
            support_episodes_by_task[task] += episodes

        self._support_episodes = defaultdict(list)
        self._support_num_episodes = 0
        for task, eps in support_episodes_by_task.items():
            if len(eps) > self._max_episodes_per_task:
                eps = random.sample(eps, self._max_episodes_per_task)
            for (task, var, ep) in eps:
                chunks = math.ceil(max_episode_length_dict[task] / self._max_episode_length)
                for chunk in range(chunks):
                    self._support_episodes[(task, chunk)].append((task, var, ep, chunk))
                    self._support_num_episodes += 1
        self._support_episodes = dict(self._support_episodes)

        print(f"Created dataset from main root {main_root} and support root {support_root} "
              f"with {self._main_num_episodes} main episodes and {self._support_num_episodes} "
              f"support episodes")

    def __getitem__(self, episode_id: int) -> Optional[Sample]:
        episode_id %= self._main_num_episodes
        task, variation, file, chunk = self._main_episodes[episode_id]

        # TODO How to deal with data augmentations in this data loader? Currently we augment
        #  both the demos and the support set during training and neither during evaluation

        main_episode = self._get_episode(task, variation, file, chunk, augment=self._training)
        support_episodes = [
            self._get_episode(task, variation, file, chunk, augment=self._training)
            for task, variation, file, chunk
            in random.sample(self._support_episodes[(task, chunk)], self._support_set_size)
        ]
        if main_episode is None or any(ep is None for ep in support_episodes):
            # Episode ID is not valid, sample another one
            # TODO Need to improve this logic to properly cover tasks with a variable number
            #  of timesteps
            episode_id = random.randint(0, self._main_num_episodes - 1)
            return self.__getitem__(episode_id)

        def collate_fn(batch: List[Dict]):
            keys = batch[0].keys()
            return {
                key: default_collate([item[key] for item in batch])
                if batch[0][key] is not None
                else None
                for key in keys
            }

        episode = collate_fn([main_episode] + support_episodes)
        return episode

    def _get_episode(self, task: str, variation: int, file: str, chunk: int, augment: bool) -> Optional[Sample]:
        episode = self._cache(file)

        if episode is None:
            return None

        frame_ids = episode[0][chunk * self._max_episode_length: (chunk + 1) * self._max_episode_length]
        num_ind = len(frame_ids)
        if num_ind == 0:
            return None
        pad_len = max(0, self._max_episode_length - num_ind)

        states: torch.Tensor = torch.stack([episode[1][i].squeeze(0) for i in frame_ids])
        if states.shape[-1] != self._image_size[1] or states.shape[-2] != self._image_size[0]:
            raise ValueError(f"{states.shape} {file}")
        pad_vec = [0] * (2 * states.dim())
        pad_vec[-1] = pad_len
        states = F.pad(states, pad_vec)

        cameras = list(episode[3][0].keys())
        assert all(c in cameras for c in self._cameras)
        index = torch.tensor([cameras.index(c) for c in self._cameras])

        states = states[:, index]
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]

        attns = torch.Tensor([])
        for i in frame_ids:
            attn_cams = torch.Tensor([])
            for cam in self._cameras:
                u, v = episode[3][i][cam]
                attn = torch.zeros((1, 1, self._image_size[0], self._image_size[1]))
                if not (u < 0 or u > self._image_size[1] - 1 or v < 0 or v > self._image_size[0] - 1):
                    attn[0, 0, v, u] = 1
                attn_cams = torch.cat([attn_cams, attn])
            attns = torch.cat([attns, attn_cams.unsqueeze(0)])
        pad_vec = [0] * (2 * attns.dim())
        pad_vec[-1] = pad_len
        attns = F.pad(attns, pad_vec)
        rgbs = torch.cat([rgbs, attns], 2)

        action = torch.cat([episode[2][i] for i in frame_ids])
        shape = [0, 0] * action.dim()
        shape[-1] = pad_len
        action = F.pad(action, tuple(shape), value=0)

        mask = torch.tensor([True] * num_ind + [False] * pad_len)

        instr: torch.Tensor = random.choice(self._instructions[task][variation])

        gripper = torch.cat([episode[4][i] for i in frame_ids])
        shape = [0, 0] * gripper.dim()
        shape[-1] = pad_len
        gripper = F.pad(gripper, tuple(shape), value=0)

        tframe_ids = torch.tensor(frame_ids)
        tframe_ids = F.pad(tframe_ids, (0, pad_len), value=-1)

        if augment:
            pcds, gripper, action = self._rotate(pcds, gripper, action, mask)
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        return {
            "frame_id": tframe_ids,
            "task": task,
            "task_id": TASK_TO_ID[task],
            "variation": variation,
            "rgbs": rgbs,
            "pcds": pcds,
            "action": action,
            "padding_mask": mask,
            "instr": instr,
            "gripper": gripper,
        }

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._main_num_episodes
