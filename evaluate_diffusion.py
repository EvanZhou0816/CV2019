import numpy as np
import accuracy_check

def evaluate_diffusion():
    ranking = np.load('~/go/src/github.com/CV2019/utils/ranking.npy').item()
    _, input_name = accuracy_check.query_import()
    name2index = np.load('../name2index.npy').item()
    index2name = np.load('../index2name.npy').item()
    rate = 0
    image_size = len(input_name)
    for input in input_name:
        input_index = name2index[input]
        output_index = ranking[input_index][0]
        output_name = index2name[output_index]
        rate += accuracy_check.accuracy_check(input, output_name)
    percent = rate/image_size

    return percent