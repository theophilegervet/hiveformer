from PIL import Image
import numpy as np

# # Load the image
# img1 = np.array(Image.open('/home/zhouxian/git/hiveformer/3_front.png'))
# img2 = np.array(Image.open('/home/zhouxian/git/hiveformer/4_front.png'))
# import IPython;IPython.embed()
# img3 = np.clip((img1.astype(float) * 0.5 + img2.astype(float) * 0.5), 0, 255).astype(np.uint8)

# Image.fromarray(img3).save('overlay.png')
# import matplotlib.pyplot as plt
# plt.imshow(img3); plt.show()

transparency= 70
background = Image.open('/home/zhouxian/git/hiveformer/4_front.png').convert("RGBA")
overlay = Image.open('/home/zhouxian/git/hiveformer/3_front_edited.png').convert("RGBA")

# Create a new image the same size as the background and overlay
composite = Image.new('RGBA', background.size)

# Extract the alpha band from the overlay image
overlay_alpha = overlay.split()[3]

# Create a new alpha band that consists of the original alpha band
# adjusted to the desired transparency
new_alpha = Image.new('L', background.size)
new_alpha.paste(overlay_alpha, mask=overlay_alpha.point(lambda p: p * transparency / 100))

# Paste the overlay image on the composite image, using the new alpha band as the mask
composite.paste(background, (0,0))
composite.paste(overlay, (0,0), mask=new_alpha)

# Save the composite image
composite.save('overlay.png')
