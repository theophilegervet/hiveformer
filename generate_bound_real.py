"""
This script computes the minimum and maximum gripper locations for
each task in the training set.
"""

import tap
from typing import List, Tuple, Optional
from pathlib import Path
import torch
import pprint
import json

from utils.utils_without_rlbench import (
    load_instructions,
    get_max_episode_length,
)
from dataset import RLBenchDataset


class Arguments(tap.Tap):
    dataset = '/home/zhouxian/git/datasets/packaged/real_tasks_train'
    out_file: str = "tasks/real_tasks_location_bounds.json"
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    instructions: Optional[Path] = "instructions_old/instructions_real.pkl"

    tasks: Tuple[str, ...] = (
        "real_reach_target",
        'real_press_stapler',
        'real_press_hand_san',
        'real_put_fruits_in_bowl',
        'real_stack_bowls',
        'real_unscrew_bottle_cap',
        'real_transfer_beans',
        'real_put_duck_in_oven',
        'real_spread_sand',
        'real_wipe_coffee',
    )
    variations: Tuple[int, ...] = (0,)


if __name__ == "__main__":
    args = Arguments().parse_args()

    bounds = {task: [] for task in args.tasks}

    for task in args.tasks:
        instruction = load_instructions(
            args.instructions, tasks=[task], variations=args.variations
        )

        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
        max_episode_length = get_max_episode_length([task], args.variations)

        dataset = RLBenchDataset(
            root=args.dataset,
            image_size=(240, 360),  # type: ignore
            taskvar=taskvar,
            instructions=instruction,
            max_episode_length=max_episode_length,
            max_episodes_per_taskvar=1000,
            cache_size=0,
            cameras=args.cameras,  # type: ignore
            training=False
        )

        print(f"Computing gripper location bounds for task {task} from dataset of "
              f"length {len(dataset)}")

        for i in range(len(dataset)):
            ep = dataset[i]
            bounds[ep["task"]].append(ep["action"][ep["padding_mask"], :3])

    bounds = {
        task: [
            torch.cat(gripper_locs, dim=0).min(dim=0).values.tolist(),
            torch.cat(gripper_locs, dim=0).max(dim=0).values.tolist()
        ]
        for task, gripper_locs in bounds.items()
        if len(gripper_locs) > 0
    }

    pprint.pprint(bounds)
    json.dump(bounds, open(args.out_file, "w"), indent=4)
