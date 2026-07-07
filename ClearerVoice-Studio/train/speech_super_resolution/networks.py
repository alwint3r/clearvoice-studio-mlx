import torch
import torch.nn as nn
import torch.nn.functional as F

class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args
        if args.network == 'MossFormer2_SR_48K':
            # Import the MossFormer2 speech enhancement model for 48 kHz
            from models.mossformer2_sr.mossformer2_sr_wrapper import MossFormer2_SR_48K
            # Initialize the model
            self.models = nn.ModuleList()
            self.models.append(MossFormer2_SR_48K(args).model_m)
            self.models.append(MossFormer2_SR_48K(args).model_g)
            self.discs = nn.ModuleList()
            self.discs.append(MossFormer2_SR_48K(args).mpd)
            self.discs.append(MossFormer2_SR_48K(args).msd)
            self.discs.append(MossFormer2_SR_48K(args).mbd)
        else:
            print("No network found!")
            return

    def forward(self, inputs, visual=None):
        latent_rep = self.model[0](inputs)
        outputs = self.model[1](latent_rep)
        return outputs
