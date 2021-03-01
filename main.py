import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from config import read_config
from layers.LightNet_plusplus import LightNet_plusplus_Model
from layers.ablation import Ablation_without_T, Ablation_without_W, Ablation_without_WandT
from generator import DataGenerator
from scores import Model_eval


def selectModel(config_dict):
    if config_dict['NetName'] == 'LightNet_plusplus':
        model = LightNet_plusplus_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_T':
        model = Ablation_without_T(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_W':
        model = Ablation_without_W(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_WandT':
        model = Ablation_without_WandT(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model


def DoTrain(config_dict):
    if not os.path.isdir(config_dict['ModelFileDir']):
        os.makedirs(config_dict['ModelFileDir'])

    if not os.path.isdir(config_dict['RecordFileDir']):
        os.makedirs(config_dict['RecordFileDir'])

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    # data index
    TrainSetFilePath = 'data_index/TrainCase.txt'
    ValSetFilePath = 'data_index/ValCase.txt'
    TestSetFilePath = 'data_index/TestCase.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            train_list.append(line.rstrip('\n').rstrip('\r\n'))
    val_list = []
    with open(ValSetFilePath) as file:
        for line in file:
            val_list.append(line.rstrip('\n').rstrip('\r\n'))
    test_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            test_list.append(line.rstrip('\n').rstrip('\r\n'))
    # data
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)
    test_data = DataGenerator(test_list, config_dict)
    test_loader = DataLoader(dataset=test_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)

    # model
    model = selectModel(config_dict)

    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])

    # eval
    model_eval_valdata = Model_eval(config_dict, is_save_model=True)
    model_eval_testdata = Model_eval(config_dict, is_save_model=False)

    print('Beginning train!')

    for epoch in range(config_dict['EpochNum']):
        for i, (X, y) in enumerate(train_loader):
            wrf, obs = X
            label = y
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            label = label.to(config_dict['Device'])

            pre_frames = model(wrf, obs)

            # backward
            optimizer.zero_grad()
            loss = criterion(torch.flatten(pre_frames), torch.flatten(label))
            loss.backward()

            # update weights
            optimizer.step()

            # output
            print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))

        val_sumets = model_eval_valdata.eval(val_loader, model, epoch)
        test_sumets = model_eval_testdata.eval(test_loader, model, epoch)
        print(val_sumets, test_sumets)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config_dict = read_config()

    # train
    DoTrain(config_dict)




