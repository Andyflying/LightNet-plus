import torch
import torch.nn as nn

class ConvLSTM2D(nn.Module):
    def __init__(self, channels, filters, kernel_size, img_rowcol):
        super(ConvLSTM2D, self).__init__()
        # self.channels = channels
        self.filters = filters
        self.padding = kernel_size // 2
        # self.kernel_size = kernel_size
        # self.strides = strides
        self.conv_x = nn.Conv2d(channels, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        self.conv_h = nn.Conv2d(filters, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=False)
        self.mul_c = nn.Parameter(torch.zeros([1, filters * 3, img_rowcol, img_rowcol], dtype=torch.float32))


    def forward(self, x, h, c):
        # x -> [batch_size, channels, x, y]
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.filters, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.filters, dim=1)
        i_c, f_c, o_c = torch.split(self.mul_c, self.filters, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c * c)
        f_t = torch.sigmoid(f_x + f_h + f_c * c)
        c_t = torch.tanh(c_x + c_h)
        c_next = i_t * c_t + f_t * c
        o_t = torch.sigmoid(o_x + o_h + o_c * c_next)
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next
