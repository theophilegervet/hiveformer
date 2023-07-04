import random

import torch

from .dataset_engine import RLBenchDataset


class RLBenchOpenLoopDataset(RLBenchDataset):
    """RLBench dataset, loads whole episodes."""

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
        predict_short=None
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
            dense_interpolation=dense_interpolation,
            interpolation_length=interpolation_length,
            action_dim=action_dim
        )

    def __getitem__(self, episode_id):
        """
        This dataset concatenates the WHOLE episode trajectory.

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

        # Keep only the first observation
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
        rgbs = self._unnormalize_rgb(rgbs)

        # Action (goal) is the last macro-action
        action = episode[2][-1]

        # Sample one instruction feature
        instr = random.choice(self._instructions[task][variation])
        instr = instr[None].repeat(len(rgbs), 1, 1)

        # Get gripper tensor for the first frame
        gripper = episode[4][0]

        # Low-level trajectory
        traj = torch.cat(episode[5])  # WHOLE episode
        traj = self._interpolate_traj(traj)[None]  # (1, L, 8)
        traj_lens = torch.as_tensor([traj.size(1)])

        # Augmentations
        if self._training:
            pcds, gripper, action, traj = self._rotate(
                pcds, gripper, action, None, traj
            )
            for t, tlen in enumerate(traj_lens):
                traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        return {
            "task": [task],  # n_frames is 1 here
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action[..., :self._action_dim],  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper[..., :self._action_dim],  # e.g. tensor (n_frames, 8), current pose
            "trajectory": traj[..., :self._action_dim],  # e.g. tensor (n_frames, 67, 8)
            "trajectory_len": traj_lens  # e.g. tensor (n_frames,)
        }
