from waveUNet.model.resample import *
import torch.nn as nn

class LinearUpsample(nn.Upsample):
    def __init__(self, *args, **kwargs):
        super(LinearUpsample, self).__init__(*args, **kwargs)
        assert(self.scale_factor is not None)

    def get_output_size(self, input_size):
        return input_size*int(self.scale_factor)

class Decimate(nn.Module):
    def __init__(self, factor):
        super(Decimate, self).__init__()
        self.factor = int(factor)

    def forward(self, x):
        return x[:,:, ::self.factor] # Decimate by factor of 2 # out = (in-1)/2 + 1

    def get_input_size(self, output_size):
        return output_size*self.factor