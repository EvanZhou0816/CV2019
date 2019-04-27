import numpy as np
import gzip
import tarfile
import os
# from raw_data import process_image
from PIL import Image
if __name__ == "__main__":
    print(os.getcwd())
    os.chdir('F:/workspace_Python/CV2019')
    tar = tarfile.open("../oxbuild_images.tgz")
    for tar_info in tar.getmembers():
        f = tar.extractfile(tar_info)
        f.read()
        imgobj = Image.open(f)
        print(imgobj)
        img = np.array(imgobj)
        print(img.shape)
        break

    pass