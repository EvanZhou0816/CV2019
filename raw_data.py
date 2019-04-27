import gzip
import numpy as np
import tarfile

#image total 5062
# def raw_data():
#     i = 0
#     tar = tarfile.open("oxbuild_images.tgz")
#     for member in tar.getmembers():
#         file = tar.extractfile(member)
#         temp = np.fromstring(file.read(), dtype=np.uint8)
#         temp /= 255
#         if not i % 250:
#             print("%d images to array" % i)
#     print("All images to array")
# raw_data()


# from scipy import misc
# rgb = np.zeros((255, 255, 3), dtype=np.uint8)
# rgb[..., 0] = np.arange(255)
# rgb[..., 1] = 55
# rgb[..., 2] = 1 - np.arange(255)
# misc.imsave('/tmp/rgb_gradient.png', rgb)
# print(rgb)

from PIL import Image
def process_image(filename):
    img = Image.open(filename)
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