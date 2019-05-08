import numpy as np
import tarfile
import fnmatch


def query_import():
    query_name = []
    query_directory = {}
    tar = tarfile.open('gt_files_170407.tgz', 'r')
    target = 'query'
    for file_name in tar.getnames():
        if target in file_name:
            file = tar.getmember(file_name)
            query_name.append(file_name)
            file_prefix = file_name[:-9]
            f = tar.extractfile(file)
            content = f.read()
            query_image = str(content).split(' ')[0][2:]
            query_directory[query_image] = file_prefix
    print(query_directory)
    return query_directory, query_name


def accuracy_check(query_image, result_image):
    query_directory = query_import()
    query_prefix = query_directory[query_image]
    tar = tarfile.open('gt_files_170407.tgz', 'r')
    good = query_prefix + 'good'
    ok = query_prefix + 'ok'
    junk = query_prefix + 'junk'
    print(good)
    print(ok)
    print(junk)
    for file_name in tar.getnames():
        if good in file_name:
            file = tar.getmember(file_name)
            f = tar.extractfile(file)
            content =  f.read()
            print(str(content))
            if result_image in str(content):
                print('goodddddd')
                rate = 1
                print(rate)
                return rate
        elif ok in file_name:
            file = tar.getmember(file_name)
            f = tar.extractfile(file)
            content = f.read()
            if result_image in str(content):
                print('okkkkkk')
                rate = 0.75
                print(rate)
                return rate
        elif junk in file_name:
            file = tar.getmember(file_name)
            f = tar.extractfile(file)
            content = f.read()
            if result_image in str(content):
                print('junkkkkk')
                rate = 0.5
                print(rate)
                return rate
        else:
            rate = 0
            print(rate)
            return rate

accuracy_check('oxc1_all_souls_000013', 'balliol_000211')

