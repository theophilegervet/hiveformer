import glob
import os
import blosc
import pickle as pkl
import numpy as np
files = glob.glob(os.path.join('/home/zhouxian/git/datasets/packaged/real_tasks_train/real_push_button+0', '*.dat'))
for file in files:
    print(file)
    data = pkl.loads(blosc.decompress(open(file, "rb").read()))

    # for i in range(len(data[2])):
    #     data[2][i] = data[2][i].unsqueeze(0)

    # for i in range(len(data[4])):
    #     data[4][i] = data[4][i].unsqueeze(0)

    data[1] = data[1].astype(np.float32)

    with open(file, "wb") as f:
        f.write(blosc.compress(pkl.dumps(data)))
