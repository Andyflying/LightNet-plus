import torch
import torch.nn as nn
from torch.nn import functional as F
# from layers.Transformer import Transformer
from layers.transformer_decoder import TransformerDecoder as Transformer
from layers.ConvLSTM import ConvLSTM2D
from deformable_convolution.modules import ModulatedDeformConvPack, ModulatedDeformConvTM

class LiteEncoder(nn.Module):
    def __init__(self, config_dict):
        super(LiteEncoder, self).__init__()
        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.conv2d_qice = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_qsnow = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_qgroup = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_w = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2d_rain = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layernorm = nn.LayerNorm([5, mn, mn], elementwise_affine=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )


    def forward(self, wrf):
        wrf_qice = wrf[:, 3:7]
        wrf_qice = torch.layer_norm(wrf_qice, normalized_shape=tuple(wrf_qice[0, 0].shape), eps=1e-30)
        wrf_qsnow = wrf[:, 12:16]
        wrf_qsnow = torch.layer_norm(wrf_qsnow, normalized_shape=tuple(wrf_qsnow[0, 0].shape), eps=1e-30)
        wrf_qgroup = wrf[:, 21:25]
        wrf_qgroup = torch.layer_norm(wrf_qgroup, normalized_shape=tuple(wrf_qgroup[0, 0].shape), eps=1e-30)
        wrf_w = wrf[:, 27:28]
        wrf_rain = wrf[:, 28:29]

        wrf_qice = self.conv2d_qice(wrf_qice)
        wrf_qsnow = self.conv2d_qsnow(wrf_qsnow)
        wrf_qgroup = self.conv2d_qgroup(wrf_qgroup)
        wrf_w = self.conv2d_w(wrf_w)
        wrf_rain = self.conv2d_rain(wrf_rain)

        wrf_enc = torch.cat([wrf_qice, wrf_qsnow, wrf_qgroup, wrf_w, wrf_rain], dim=1)
        wrf_enc = self.layernorm(wrf_enc)
        wrf_enc = self.encoder(wrf_enc)
        return wrf_enc


class WRFInfo(nn.Module):
    def __init__(self, channels, config_dict):
        super(WRFInfo, self).__init__()
        assert channels % 2 == 0
        self.config_dict = config_dict
        self.channels = channels
        self.wrf_encoder_convLSTM2D_for = ConvLSTM2D(channels, channels//2, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.wrf_encoder_convLSTM2D_rev = ConvLSTM2D(channels, channels//2, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)

    def forward(self, wrf):
        # wrf : [frames, batch_size, channels, x, y]
        batch_size = wrf.shape[1]
        wrf_h_alltime_for = torch.zeros([self.config_dict['ForecastHourNum'], batch_size, self.channels // 2,
                           wrf.shape[3], wrf.shape[4]], dtype=torch.float32).to(wrf.device)
        wrf_h_alltime_rev = torch.zeros([self.config_dict['ForecastHourNum'], batch_size, self.channels // 2,
                                         wrf.shape[3], wrf.shape[4]], dtype=torch.float32).to(wrf.device)
        # forward LSTM
        wrf_h = torch.zeros([batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], dtype=torch.float32).to(wrf.device)
        wrf_c = torch.zeros([batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], dtype=torch.float32).to(wrf.device)
        for i in range(self.config_dict['ForecastHourNum']):
            wrf_h_alltime_for[i] = wrf_h
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D_for(wrf[i], wrf_h, wrf_c)

        # reverse LSTM
        wrf_h = torch.zeros([batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], dtype=torch.float32).to(wrf.device)
        wrf_c = torch.zeros([batch_size, self.channels//2, wrf.shape[3], wrf.shape[4]], dtype=torch.float32).to(wrf.device)
        for i in range(self.config_dict['ForecastHourNum']-1, 0, -1):
            wrf_h_alltime_rev[i] = wrf_h
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D_rev(wrf[i], wrf_h, wrf_c)

        # concat
        wrf_h_alltime = torch.cat([wrf_h_alltime_for, wrf_h_alltime_rev], dim=2)
        return wrf_h_alltime, None


class Ablation_without_T(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super(Ablation_without_T, self).__init__()
        self.config_dict = config_dict
        self.obs_tra_frames = obs_tra_frames
        self.wrf_tra_frames = wrf_tra_frames
        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        self.encoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.wrf_encoder_module = LiteEncoder(config_dict=config_dict)
        self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.dcn_tm = ModulatedDeformConvTM(8, 8, kernel_size=1, stride=1, padding=0)
        self.decoder_module = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1)
        )
        self.conv_fusion_h = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2, groups=2)
        self.wrf_info = WRFInfo(channels=8, config_dict=config_dict)

    def forward(self, wrf, obs):
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()
        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()

        batch_size = obs.shape[1]
        pre_frames = torch.zeros([self.wrf_tra_frames, batch_size, 1, wrf.shape[3], wrf.shape[4]]).to(wrf.device)

        h = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        # WRF info encoder
        wrf_encoder = []
        for t in range(self.wrf_tra_frames):
            wrf_encoder.append(self.wrf_encoder_module(wrf[t]))
        wrf_encoder = torch.stack(wrf_encoder, dim=0)
        wrf_info_h, _ = self.wrf_info(wrf_encoder)

        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        h = self.encoder_h(h)
        c = self.encoder_c(c)
        for t in range(self.wrf_tra_frames):
            his_enc = self.conv_fusion_h(torch.cat([wrf_info_h[t], h], dim=1))
            h, c = self.decoder_ConvLSTM(his_enc, h, c)
            pre = self.dcn_tm(h, t + 1)
            pre = self.decoder_module(pre)
            pre_frames[t] = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        return pre_frames


class Ablation_without_W(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super(Ablation_without_W, self).__init__()
        self.config_dict = config_dict
        self.obs_tra_frames = obs_tra_frames
        self.wrf_tra_frames = wrf_tra_frames
        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        self.encoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.wrf_encoder_module = LiteEncoder(config_dict=config_dict)
        self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.dcn_tm = ModulatedDeformConvTM(8, 8, kernel_size=1, stride=1, padding=0)
        self.decoder_module = nn.Sequential(
            # ModulatedDeformConvPack(8, 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1)
        )
        self.conv_fusion_h = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2, groups=2)
        self.transformer = Transformer(channels=8, layers=1, nhead=1, is_posemb=True)

    def forward(self, wrf, obs):
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()
        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()

        batch_size = obs.shape[1]
        pre_frames = torch.zeros([self.wrf_tra_frames, batch_size, 1, wrf.shape[3], wrf.shape[4]]).to(wrf.device)

        h = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)

        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        h = self.encoder_h(h)
        c = self.encoder_c(c)
        for t in range(self.wrf_tra_frames):
            wrf_encoder = self.wrf_encoder_module(wrf[t])
            wrf_tf = self.transformer(wrf_encoder, h)
            h, c = self.decoder_ConvLSTM(wrf_tf, h, c)
            pre = self.dcn_tm(h, t + 1)
            pre = self.decoder_module(pre)
            pre_frames[t] = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        return pre_frames


class Ablation_without_WandT(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels, config_dict):
        super(Ablation_without_WandT, self).__init__()
        self.config_dict = config_dict
        self.obs_tra_frames = obs_tra_frames
        self.wrf_tra_frames = wrf_tra_frames
        mn = (config_dict['GridRowColNum'] // 2) // 2
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        self.encoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.wrf_encoder_module = LiteEncoder(config_dict=config_dict)
        self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)
        self.dcn_tm = ModulatedDeformConvTM(8, 8, kernel_size=1, stride=1, padding=0)
        self.decoder_module = nn.Sequential(
            # ModulatedDeformConvPack(8, 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1, stride=1)
        )
        self.conv_fusion_h = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2, groups=2)

    def forward(self, wrf, obs):
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()
        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()

        batch_size = obs.shape[1]
        pre_frames = torch.zeros([self.wrf_tra_frames, batch_size, 1, wrf.shape[3], wrf.shape[4]]).to(wrf.device)

        h = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)

        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        h = self.encoder_h(h)
        c = self.encoder_c(c)
        for t in range(self.wrf_tra_frames):
            wrf_encoder = self.wrf_encoder_module(wrf[t])
            h, c = self.decoder_ConvLSTM(wrf_encoder, h, c)
            pre = self.dcn_tm(h, t + 1)
            pre = self.decoder_module(pre)
            pre_frames[t] = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        return pre_frames

if __name__ == "__main__":
    pass

