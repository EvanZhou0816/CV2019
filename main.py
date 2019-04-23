import numpy as np
import gzip
import tarfile
import os


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir('F:/workspace_Python/CV2019')
    tar = tarfile.open("./oxbuild_images.tgz")
    for tar_info in tar.getmembers():
        f = tar.extractfile(tar_info)
        img = f.read()
        print(img)

    print(f) 
    # with open("./oxbuild_images.tgz",'rb') as f:
    #     print(np.array(f).shape)
    pass