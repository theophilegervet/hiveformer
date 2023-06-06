import numpy as np
from PIL import Image

# Load the .npy file using NumPy
start_np = np.load('/home/zhouxian/Downloads/figure_materials/start_kinect_rgb.npy')
end_np = np.load('/home/zhouxian/Downloads/figure_materials/end_kinect_rgb.npy')

# Convert the NumPy array to PIL image
start = Image.fromarray(start_np)
end = Image.fromarray(end_np)

start.save('start.png')
end.save('end.png')


# overlay
start_overlay = np.zeros_like(start_np)
start_overlay[0:500, 300:570] = start_np[0:500, 300:570].astype(np.uint8)
for i in range(720):
    for j in range(1080):
        if i + j > 880:
            start_overlay[i, j] = 0

# start_overlay[380:500, 500:570] = 0
# start_overlay[450:500, 420:570] = 0
# start_overlay[410:500, 460:570] = 0

overlayed = np.zeros_like(start_np)
for i in range(720):
    for j in range(1080):
        if start_overlay[i, j].sum() > 0:
            overlayed[i, j] = (start_overlay[i, j] * 0.5 + end_np[i, j] * 0.5).astype(np.uint8)
        else:
            overlayed[i, j] = end_np[i, j]
Image.fromarray(overlayed).show()