import sys
import os
import numpy as np
import glob


def readFileForm():
    d = {}
    with open('dataset/result.txt') as f:
        for line in f:
            key = line.split(' ')[0]
            writer = line.split(' ')[1]
            d[key] = writer
    return d


def getImageFiles():
    tmp = []
    path_to_files = os.path.join('dataset/data_subset', '*')
    for filename in sorted(glob.glob(path_to_files)):
        tmp.append(filename)
    return tmp


def getImageTargets(dic):
    target_list = []
    path_to_files = os.path.join('dataset/data_subset', '*')
    for filename in sorted(glob.glob(path_to_files)):
        image_name = filename.split('/')[-1]
        file, ext = os.path.splitext(image_name)
        parts = file.split('-')
        form = parts[0] + '-' + parts[1]
        for key in dic:
            if key == form:
                target_list.append(str(dic[form]))
    return target_list


def main() -> int:
    dic = readFileForm()
    img_files = np.asarray(getImageFiles())
    img_targets = np.asarray(getImageTargets(dic))

    print(img_files.shape)
    print(img_targets.shape)
    return 0


if __name__ == '__main__':
    sys.exit(main())
