# -*- coding: utf-8 -*-
import numpy as np
from netCDF4 import Dataset
from torch.utils.data import Dataset as py_Dataset
import datetime
import os
import scipy.sparse as sp


def getTimePeriod(dt):
    time = dt.strftime("%H:%M:%S")
    hour = int(time[0:2])
    if 0 <= hour < 6:
        nchour = '00'
    elif 6 <= hour < 12:
        nchour = '06'
    elif 12 <= hour < 18:
        nchour = '12'
    elif 18 <= hour <= 23:
        nchour = '18'
    else:
        print('error')
    delta_hour = hour - int(nchour)
    return nchour, delta_hour

def getHoursGridFromNPY(filepath, delta_hour, config_dict):  # 20200619
    # m = config_dict['GridRowColNum']
    # n = config_dict['GridRowColNum']
    grid_list = []
    param_list = ['QICE_ave3', 'QSNOW_ave3', 'QGRAUP_ave3', 'W_max', 'RAINNC']
    for s in param_list:
        npy_grid = np.load(os.path.join(filepath, '{}.npy'.format(s)))
        npy_grid = npy_grid[delta_hour:delta_hour + config_dict['ForecastHourNum']]
        if s == 'RAINNC':
            npy_grid = npy_grid[:, np.newaxis, :, :]
        elif s == 'W_max':
            npy_grid = np.max(npy_grid, axis=1, keepdims=True)
        npy_grid = np.transpose(npy_grid, (0, 2, 3, 1))  # (12, 159, 159, x)
        grid_list.append(npy_grid)
    grid = np.concatenate(grid_list, axis=-1)
    return grid


class DataGenerator(py_Dataset):
    def __init__(self, lists, config_dict):
        self.lists = lists
        self.config_dict = config_dict

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        X, y = self.__data_generation(self.lists[index])
        return X, y

    def __data_generation(self, datetime_peroid):
        m = self.config_dict['GridRowColNum']
        n = self.config_dict['GridRowColNum']
        label_batch = np.zeros(shape=[self.config_dict['ForecastHourNum'], m * n, 1], dtype=np.float32)
        history_batch = np.zeros(shape=[self.config_dict['TruthHistoryHourNum'], m, n, 1], dtype=np.float32)

        ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
        # read WRF
        utc = ddt + datetime.timedelta(hours=-8)
        ft = utc + datetime.timedelta(hours=(-6))
        nchour, delta_hour = getTimePeriod(ft)
        delta_hour += 6

        filepath = os.path.join(self.config_dict['WRFFileDir'], ft.date().strftime("%Y%m%d"), str(nchour))

        wrf_batch = getHoursGridFromNPY(filepath, delta_hour, self.config_dict)

        # read labels
        for hour_plus in range(self.config_dict['ForecastHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            tFilePath = self.config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '.npy'
            truth_grid = np.load(tFilePath)
            truth_grid[truth_grid > 1] = 1
            truth_grid = truth_grid.flatten()
            label_batch[hour_plus, :, :] = truth_grid[:, np.newaxis]
        # read history observations
        for hour_plus in range(self.config_dict['TruthHistoryHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus - self.config_dict['TruthHistoryHourNum'])
            tFilePath = self.config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '.npy'
            truth_grid = np.load(tFilePath)
            truth_grid = truth_grid.reshape(m, n)
            history_batch[hour_plus, :, :, :] = truth_grid[:, :, np.newaxis]
        return [wrf_batch, history_batch], label_batch


if __name__ == "__main__":
    from config import read_config
    from torch.utils.data import DataLoader
    config_dict = read_config()
    # chooseWindChannels(config_dict)
