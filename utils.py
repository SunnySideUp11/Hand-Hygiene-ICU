import numpy as np
import pandas as pd
import cv2
import os


def get_frame(file_path, save_folder):
    video = cv2.VideoCapture(file_path)
    count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        save_path = os.path.join(save_folder, f'{count}.jpg')
        cv2.imwrite(save_path, frame)
        count += 1
    video.release()


def get_csv_data(path, left=True):
    """
    Args:
        path (str): data path
        left (bool, optional): Whether it is left-handed or not
    """
    df = pd.read_csv(path)
    x = "Left" if left else "Right"
    csv_data = [] 
    for item in df[x]:
        data = item[1: -1]
        if len(data) == 0:
            csv_data.append([0 for _ in range(63)])
        else:
            data = [float(i) for i in data.split(", ")]
            csv_data.append(data)
    return csv_data


def get_person_data(view_idx, person_idx):
    path = "../data/landmark/view{v_i}/{p_idx}/{action_idx}.csv"
    _x, _y =[], []
    for i in range(1, 8):
        left = get_csv_data(path.format(v_i=view_idx, p_idx=person_idx, action_idx=i), left=True)
        right = get_csv_data(path.format(v_i=view_idx, p_idx=person_idx, action_idx=i), left=False)
        x = np.concatenate([left, right], axis=1)
        y = np.array([i for _ in range(len(left))])
        _x.append(x)
        _y.append(y)
    return np.concatenate(_x), np.concatenate(_y)

def get_generalized_data(view_idx):
    X = [get_person_data(view_idx=view_idx, person_idx=i)[0] for i in range(1, 9)]
    Y = [get_person_data(view_idx, i)[1] for i in range(1, 9)]
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y

def get_personalized_data(view_idx=1):
    data = []
    for i in range(1, 9):
        data.append(get_person_data(view_idx=view_idx, person_idx=i))

    personalized_data = []
    for item in data:
        copy_data = data.copy()
        copy_data.remove(item)
        _x, _y = [], []
        for x, y in copy_data:
            _x.append(x)
            _y.append(y)
        _x, _y = np.concatenate(_x), np.concatenate(_y)
        data_train = (_x, _y)
        data_test = (item[0], item[1])
        personalized_data.append([data_train, data_test])
    return personalized_data
