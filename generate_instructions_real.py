import torch
tasks = [
'real_reach_target'
]

ins = dict()
for task in tasks:
    ins[task] = dict()
    ins[task][0] = torch.zeros([1,53,512])

import pickle as pkl
pkl.dump(ins, open('/home/zhouxian/git/hiveformer/instructions_old/instructions_real.pkl', 'wb'))