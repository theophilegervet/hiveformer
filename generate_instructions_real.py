import torch
tasks = [
    'real_reach_target',
    'real_press_stapler',
    'real_press_hand_san',
    'real_put_fruits_in_bowl',
    'real_stack_bowls',
    'real_unscrew_bottle_cap',
    'real_transfer_beans',
    'real_put_duck_in_oven',
    'real_spread_sand',
    'real_wipe_coffee',
]

ins = dict()
for task in tasks:
    ins[task] = dict()
    ins[task][0] = torch.zeros([1,53,512])

import pickle as pkl
pkl.dump(ins, open('/home/zhouxian/git/hiveformer/instructions_old/instructions_real.pkl', 'wb'))