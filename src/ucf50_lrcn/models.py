"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the
              fully-connected layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a
          final fully-connected layer to produce class logits.
"""

import torch
from torch import nn
from torchvision import models

# Supported backbones: name -> (builder, ImageNet-1k classification weights).
BACKBONES = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.

    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.

    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame
    features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet.
                                     Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                   'resnet152'.
                                   Default is 'resnet34'.
        temporal_pool (str, optional): 'last' classifies from the final LSTM step; 'mean'
                                       classifies every step and averages the logits over time,
                                       as in the original LRCN. Default is 'last'.
        in_channels (int, optional): Input channels per frame; 3 for RGB, 2 * FLOW_STACK for
                                     stacked flow. Default is 3.

    Raises:
        ValueError: If the specified cnn_model or temporal_pool is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True,
                 cnn_model='resnet34', temporal_pool='last', in_channels=3):
        super().__init__()
        if temporal_pool not in ('last', 'mean'):
            raise ValueError("temporal_pool must be 'last' or 'mean'.")
        self.temporal_pool = temporal_pool

        # Set up the ResNet backbone as a 2D CNN feature extractor. pretrained=True loads the
        # ImageNet-1k classification weights (IMAGENET1K_V1, what the deprecated pretrained=True
        # used to load); no video or detection pretraining is involved.
        if cnn_model not in BACKBONES:
            raise ValueError('The input CNN backbone is not supported, please choose a valid '
                             'ResNet variant.')
        builder, weights = BACKBONES[cnn_model]
        base_cnn = builder(weights=weights if pretrained else None)
        if in_channels != 3:
            # Cross-modality initialisation for flow input: average the pretrained RGB filters
            # over colour and copy the result to every input channel.
            old = base_cnn.conv1
            base_cnn.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                                       old.stride, old.padding, bias=False)
            with torch.no_grad():
                base_cnn.conv1.weight.copy_(
                    old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features. Inputs arrive as
        # (batch, time, features), so batch_first must be set; without it the LSTM runs
        # along the batch dimension and mixes unrelated videos.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)

        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)

        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.

        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)

        All frames of all clips go through the CNN backbone in one call, so BatchNorm sees
        batch_size * time_steps images rather than batch_size. The LSTM then runs over the whole
        sequence of frame features. With temporal_pool='last' the output of the final time step
        goes through dropout and the fully-connected layer; with 'mean' every time step does, and
        the logits are averaged over time.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        features = self.base_model(x.reshape(bs * ts, c, h, w)).reshape(bs, ts, -1)
        out, _ = self.rnn(features)

        if self.temporal_pool == 'mean':
            return self.fc(self.dropout(out)).mean(dim=1)
        return self.fc(self.dropout(out[:, -1]))
