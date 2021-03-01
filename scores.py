import torch
import datetime
import os

class Cal_params_epoch(object):
    def __init__(self):
        # n1->TP  n2->FP  n3->FN  n4->TN
        self.n1 = 0
        self.n2 = 0
        self.n3 = 0
        self.n4 = 0
        self.n1sum = 0
        self.n2sum = 0
        self.n3sum = 0
        self.n4sum = 0
        self.eps = 1e-10

    def _transform_sum(self, y_true, y_pred):
        # y_true = y_true.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
        y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_pred = torch.round(torch.sigmoid(y_pred))
        frames = y_true.shape[0]
        sum_true = torch.zeros(y_true[0].shape)
        sum_pred = torch.zeros(y_pred[0].shape)
        for i in range(frames):
            sum_true += y_true[i]
            sum_pred += y_pred[i]
        sum_true = torch.flatten(sum_true)
        sum_pred = torch.flatten(sum_pred)
        return sum_true, sum_pred

    def _transform(self, y_true, y_pred):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(torch.sigmoid(y_pred))
        y_pred = torch.round(y_pred)
        return y_true, y_pred

    def _POD_(self, n1, n3):
        return torch.true_divide(n1, n1 + n3 + self.eps)

    def _FAR_(self, n1, n2):
        return torch.true_divide(n2, n1 + n2 + self.eps)

    def _TS_(self, n1, n2, n3):
        return torch.true_divide(n1, n1 + n2 + n3 + self.eps)

    def _ETS_(self, n1, n2, n3, r):
        return torch.true_divide(n1 - r, n1 + n2 + n3 - r + self.eps)

    def cal_batch(self, y_true, y_pred):
        y_true, y_pred = self._transform(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        self.n1 += n1
        self.n2 += n2
        self.n3 += n3
        self.n4 += n4
        return pod, far, ts, ets

    def cal_batch_sum(self, y_true, y_pred):
        y_true, y_pred = self._transform_sum(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        self.n1sum += n1
        self.n2sum += n2
        self.n3sum += n3
        self.n4sum += n4
        return pod, far, ts, ets

    def cal_epoch(self):
        r = torch.true_divide((self.n1 + self.n2) * (self.n1 + self.n3), self.n1 + self.n2 + self.n3 + self.n4)
        pod = self._POD_(self.n1, self.n3)
        far = self._FAR_(self.n1, self.n2)
        ts = self._TS_(self.n1, self.n2, self.n3)
        ets = self._ETS_(self.n1, self.n2, self.n3, r)
        return pod, far, ts, ets

    def cal_epoch_sum(self):
        r = torch.true_divide((self.n1sum + self.n2sum) * (self.n1sum + self.n3sum), self.n1sum + self.n2sum + self.n3sum + self.n4sum)
        pod = self._POD_(self.n1sum, self.n3sum)
        far = self._FAR_(self.n1sum, self.n2sum)
        ts = self._TS_(self.n1sum, self.n2sum, self.n3sum)
        ets = self._ETS_(self.n1sum, self.n2sum, self.n3sum, r)
        return pod, far, ts, ets

class Model_eval(object):
    def __init__(self, config_dict, is_save_model):
        self.config_dict = config_dict
        self.is_save_model = is_save_model
        self.maxPOD = -0.5
        self.maxPOD_epoch = -1
        self.minFAR = 1.1
        self.minFAR_epoch = -1
        self.maxETS = -0.5
        self.maxETS_epoch = -1
        if self.is_save_model:
            with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
                f.write(str(datetime.datetime.now()) + '\r\n')
                f.write(str(config_dict) + '\r\n')

    def __del__(self):
        info = '`model name: {}`\nmaxPOD: {} maxPOD_epoch: {}\nminFAR: {} minFAR_epoch: {}\nmaxETS: {} maxETS_epoch: {}\n'\
            .format(self.config_dict['NetName'], self.maxPOD, self.maxPOD_epoch, self.minFAR, self.minFAR_epoch, self.maxETS, self.maxETS_epoch)
        print(info)
        if self.is_save_model and self.maxPOD_epoch != -1 and self.minFAR_epoch != -1 and self.maxETS_epoch != -1:
            with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
                f.write(info + '\r\n')

    def eval(self, dataloader, model, epoch):
        val_calparams_epoch = Cal_params_epoch()
        for i, (X, y) in enumerate(dataloader):
            wrf, obs = X
            label = y
            wrf = wrf.to(self.config_dict['Device'])
            obs = obs.to(self.config_dict['Device'])
            label = label.to(self.config_dict['Device'])

            pre_frames = model(wrf, obs)

            # output
            pod, far, ts, ets = val_calparams_epoch.cal_batch(label, pre_frames)
            sumpod, sumfar, sumts, sumets = val_calparams_epoch.cal_batch_sum(label, pre_frames)
            info = 'epoch:{} ({}/{})'.format(epoch, i + 1, len(dataloader))
            print(info)
        sumpod, sumfar, sumts, sumets = val_calparams_epoch.cal_epoch_sum()
        info = '`{}` VAL EPOCH INFO: epoch:{} \nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n save model:{}\n'. \
            format(self.config_dict['NetName'], epoch, sumpod, sumfar, sumts, sumets, self.is_save_model)
        print(info)
        with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
            f.write(info + '\r\n')
        if self.is_save_model:
            if sumpod > self.maxPOD:
                self.maxPOD = sumpod
                self.maxPOD_epoch = epoch
                self.save_model(model, 'model_maxPOD', epoch)
            if 1e-6 < sumfar < self.minFAR:
                self.minFAR = sumfar
                self.minFAR_epoch = epoch
                self.save_model(model, 'model_minFAR', epoch)
            if sumets > self.maxETS:
                self.maxETS = sumets
                self.maxETS_epoch = epoch
                self.save_model(model, 'model_maxETS', epoch)
            self.save_model(model, 'new_model', epoch)
        return sumets

    def save_model(self, model, name, epoch):
        torch.save(model.state_dict(), os.path.join(self.config_dict['ModelFileDir'], '{}_{}.pkl'.format(self.config_dict['NetName'], name)))
        info = 'save model file: {} successflly! (epoch={})'.format(name, epoch)
        print(info)
        # with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
        #     f.write(info + '\r\n')

if __name__ == "__main__":
    pass