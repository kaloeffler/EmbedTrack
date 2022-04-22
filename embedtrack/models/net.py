"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import embedtrack.models.BranchedERFNet as BranchedERFNet
import torch
from torch import nn
from embedtrack.models.erfnet import Decoder


class TrackERFNet(BranchedERFNet):
    """
    Tracking network. Consists of a single, shared encoder and 3 distinct decoders.
    2 decoders are trained on segmentation, whereas 1 decoder is trained on tracking.
    """

    def __init__(self, n_classes=[4, 1, 2], input_channels=1, encoder=None):
        """
        Initialize tracking net.
        Args:
            n_classes (list): number of output channels for the 3 decoders
            (the first two inputs are the number of output channels for the segmentation decoders,
            the last one for the tracking decoder)
            input_channels (int): number of input channels
            encoder (nn.Module, optional): provide a custom encoder, otherwise an ERFNet encoder is used
        """
        super(TrackERFNet, self).__init__(
            n_classes,
            input_channels=input_channels,
            encoder=encoder,
        )
        self.decoders = nn.ModuleList()
        n_init_features = [128] * len(n_classes)
        n_init_features[-1] = 2 * n_init_features[-1]
        for n, n_feat in zip(n_classes, n_init_features):
            self.decoders.append(Decoder(n, n_init_features=n_feat))

    def forward(self, curr_frames, prev_frames):
        """
        Forward pairs of images (t, t+1). Two images in the same batch dimension position
        from the tensors curr_frames and prev_frames form an image pair from time points t and t-1.
        Args:
            curr_frames (torch.Tensor): tensor of images BxCxHxW
            prev_frames (torch.Tensor): tensor of images BxCxHxW

        Returns:

        """
        # dual input with shared weights, concat predictions -> then 2 decoders (->segmentation, flow prediction)
        images = torch.cat([curr_frames, prev_frames], dim=0)
        features_encoder = self.encoder(images)
        features_curr_frames = features_encoder[: curr_frames.shape[0]]
        features_prev_frames = features_encoder[curr_frames.shape[0] :]
        features_stacked = torch.cat(
            [features_curr_frames, features_prev_frames], dim=1
        )  # batchsize x 2 x h x w
        segm_prediction = torch.cat(
            [decoder.forward(features_encoder) for decoder in self.decoders[:-1]], 1
        )
        segm_prediction_curr = segm_prediction[: curr_frames.shape[0]]
        segm_prediction_prev = segm_prediction[curr_frames.shape[0] :]
        tracking_prediction = self.decoders[-1].forward(features_stacked)
        return segm_prediction_curr, segm_prediction_prev, tracking_prediction

    def init_output(self, n_sigma=1):
        # init last layers for tracking and segmentation offset similar
        with torch.no_grad():
            for decoder in (self.decoders[0], self.decoders[-1]):
                output_conv = decoder.output_conv
                print("Initialize last layer with size: ", output_conv.weight.size())
                print("*************************")
                output_conv.weight[:, 0:2, :, :].fill_(0)
                output_conv.bias[0:2].fill_(0)

                output_conv.weight[:, 2 : 2 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 : 2 + n_sigma].fill_(1)


def calc_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
