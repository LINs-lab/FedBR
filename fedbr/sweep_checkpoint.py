from genericpath import isdir, isfile
import os

def del_files(path):
    if os.path.isfile(path) and '.pkl' in path:
        os.remove(path)
    elif os.path.isdir(path):
        lists = os.listdir(path)
        for list in lists:
            tf = os.path.join(path, list)
            del_files(tf)

del_files('./')