import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

EPS = 1e-8


class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args

        # audio backbone network
        if args.network_audio.backbone == 'av_skim':
            from models.av_skim.av_skim import av_skim
            self.av_skim = av_skim(args)
            self._define_lip_ref_encoder()
        else:
            raise NameError('Wrong network selection')

    def _define_lip_ref_encoder(self):
        # reference network for lip encoder
        assert self.args.network_reference.cue == 'lip'

        if self.args.network_reference.backbone == 'blazenet64':
            from models.visual_frontend.blazenet64 import visualNet as Visual_encoder
        else:
            raise NameError('Wrong reference network selection')
        self.network_v = Visual_encoder(self.args)


    def forward(self, mixture, ref=None, reference=None):
        if self.args.network_audio.backbone in ['av_skim']:
            # speaker extraction with lip reference
            visual = ref.to(self.args.device)

            visual = transforms.functional.rgb_to_grayscale(visual.permute((0, 1, 4, 2, 3))).squeeze(2)
            ymin, ymax, xmin, xmax = 15, 91, 27, 103
            visual = visual[:,:,ymin:ymax, xmin:xmax]
            visual = visual.clone().detach()

            visual = self.network_v(visual)
            return self.av_skim(mixture, visual, reference)
        else:
            raise NameError('Wrong network selection')



