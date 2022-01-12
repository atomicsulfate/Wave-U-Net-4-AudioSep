from waveUNet.model.resample import *
import torch.nn as nn

class LinearUpsample(nn.Upsample):
    def __init__(self):
        super(LinearUpsample, self).__init__(mode='linear', align_corners=True)

    def forward(self, input):
        self.size = 2 * input.shape[2] - 1
        return super(LinearUpsample, self).forward(input)

    def get_output_size(self, input_size):
        return input_size*2 - 1

class Decimate(nn.Module):
    def __init__(self):
        super(Decimate, self).__init__()

    def forward(self, input):
        return input[:,:, ::2] # Decimate by factor of 2 # out = (in-1)/2 + 1

    def get_input_size(self, output_size):
        return output_size*2 - 1