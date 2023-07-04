import math
import random

import torch

from .dataset_engine import RLBenchDataset


class RLBenchShortTermDataset(RLBenchDataset):
    """RLBench dataset, loads short-term trajectories."""

    def __init__(
        self,
        # required
        root,
        instructions,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        gripper_loc_bounds=None,
        image_rescale=(1.0, 1.0),
        point_cloud_rotate_yaw_range=0.0,
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        action_dim=8,
        predict_short=16
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=cache_size,
            max_episodes_per_task=max_episodes_per_task,
            num_iters=num_iters,
            cameras=cameras,
            training=training,
            gripper_loc_bounds=gripper_loc_bounds,
            image_rescale=image_rescale,
            point_cloud_rotate_yaw_range=point_cloud_rotate_yaw_range,
            return_low_lvl_trajectory=True,
            dense_interpolation=False,
            interpolation_length=interpolation_length,
            action_dim=action_dim
        )
        self._predict_short = predict_short

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None
        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Convert the dataset to short-term trajectories of desired length
        assert self._predict_short is not None
        # First, concatenate all trajectories from current to end
        episode[-1] = [
            torch.cat(episode[-1][e:])[:self._predict_short]  # fixed length
            for e in range(len(episode[-1]))
        ]
        for e in range(len(episode[-1])):
            # For shorter trajectories, pad with last action
            if len(episode[-1][e]) < self._predict_short:
                cat_ = episode[-1][e][-1][None]
                for _ in range(self._predict_short - len(episode[-1][e])):
                    episode[-1][e] = torch.cat((episode[-1][e], cat_))
            # Change goal action to be the last action of this segment
            episode[2][e] = episode[-1][e][-1][None]

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            torch.from_numpy(episode[1][i]) for i in frame_ids
        ])

        # Camera ids
        cameras = list(episode[3][0].keys())
        assert all(c in cameras for c in self._cameras)
        index = torch.tensor([cameras.index(c) for c in self._cameras])

        # Re-map states based on camera ids
        states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        instr = random.choice(self._instructions[task][variation])
        instr = instr[None].repeat(len(rgbs), 1, 1)

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # Low-level trajectory
        traj_items = [episode[5][i] for i in frame_ids]
        max_l = max(len(item) for item in traj_items)
        traj = torch.zeros(len(traj_items), max_l, 8)
        traj_lens = torch.as_tensor(
            [len(item) for item in traj_items]
        )
        for i, item in enumerate(traj_items):
            traj[i, :len(item)] = item

        # Augmentations
        if self._training:
            pcds, gripper, action, traj = self._rotate(
                pcds, gripper, action, None, traj
            )
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        return {
            "task": [task for i in frame_ids],
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
