import numpy as np

#image total 5062
from PIL import Image
def load_image(filename):
    img = Image.open(filename)
    return img
    pass
def process_image(img):   
    # Image.frombytes()
    if img.width > 600 or img.height > 600:
        w_ratio = img.width/600
        h_ratio = img.height/600
        if w_ratio > h_ratio:
            scale = w_ratio
        else:
            scale = h_ratio
        # https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html
        # https://zhuanlan.zhihu.com/p/27504020
        img = img.resize((int(img.width/scale), int(img.height/scale)), Image.ANTIALIAS)
    # img.save('new-'+filename)
    # img.close()
    arr = np.array(img)
    res = arr/255
    return res

# process_image("all_souls_000000.jpg")