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


def readFileImage():
    tmp = []
    target_list = []
    path_to_files = os.path.join('dataset/data_subset', '*')
    d = readFileForm()
    for filename in sorted(glob.glob(path_to_files)):
        tmp.append(filename)
        image_name = filename.split('/')[-1]
        file, ext = os.path.splitext(image_name)
        parts = file.split('-')
        form = parts[0] + '-' + parts[1]
        for key in d:
            if key == form:
                target_list.append(str(d[form]))
    img_files = np.asarray(tmp)
    img_targets = np.asarray(target_list)
    print(img_files.shape)
    print(img_targets.shape)


def main() -> int:
    readFileImage()
    return 0


if __name__ == '__main__':
    sys.exit(main())
