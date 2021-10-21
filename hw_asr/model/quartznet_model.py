from torch import nn, Tensor
from torch.nn import Sequential
from hw_asr.base import BaseModel


class QuartzBaseModule(nn.Module):
    """
        Conv-BN module
    """
    def __init__(
            self,
            c_in: int,
            c_out: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = None
    ):
        super(QuartzBaseModule, self).__init__()
        if not padding:
            padding = kernel_size // 2
        self.module = Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, stride=stride, groups=c_in, padding=padding),
            nn.Conv1d(c_in, c_out, 1),
            nn.BatchNorm1d(c_out)
        )

    def forward(self, input):
        return self.module(input)


class QuartzBBlock(nn.Module):
    """
        B block in Quartznet
    """
    def __init__(
            self,
            c_in: int = None,
            c_out: int = 256,
            kernel_size: int = 33,
            n_cells: int = 5,
            dropout: float = 0.2):

        super(QuartzBBlock, self).__init__()

        self.n_cells = n_cells
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = nn.ReLU()

        blocks = [QuartzBaseModule(self.c_in, self.c_out, self.kernel_size)]
        blocks.extend(
            [
                self.activation,
                QuartzBaseModule(self.c_in, self.c_out, self.kernel_size),
            ] * (n_cells - 1)
        )
        self.blocks = Sequential(*blocks)
        self.res_block = Sequential(
            nn.Conv1d(self.c_in, self.c_out, 1),
            nn.BatchNorm1d(self.c_out)
        )

    def forward(self, input: Tensor):
        outputs = self.blocks(input)
        return self.activation(outputs + self.res_block(input))


class QuartzNet(BaseModel):
    def __init__(
            self,
            n_feats: int,
            n_class: int,
            n_bblocks: int = 1,    # S parameter (how many times a B block is repeated)
            n_bmodules: int = 5,    # R parameter (how many times a module in each B block is repeated)
            *args, **kwargs):

        """
            n_feats -- number of features (mels)
            n_class -- number of classes
            n_bblocks -- how many times B block is repeated
                1 -- QN 5x5
                2 -- QN 10x5
                3 -- QN 15x5
            n_bmodules -- how many times a module is repeated
        """

        super().__init__(n_feats, n_class, *args, **kwargs)
        self.n_feats = n_feats
        self.n_class = n_class
        self.n_bblocks = n_bblocks
        self.n_bmodules = n_bmodules
        self.activation = nn.ReLU()
        self.b_kernel_sizes = [33, 39, 51, 63, 75]
        self.b_channels = [256, 256, 512, 512, 512]
        self.c_kernel_sizes = [33, 87, 1]    # for c1-3
        self.c_channels = [256, 512, 1024]    # for c1-3

        # Conv-BN-ReLU-1: stride=2
        self.conv1 = Sequential(
            QuartzBaseModule(self.n_feats, self.c_channels[0], self.c_kernel_sizes[0], 2),
            self.activation
        )
        # B blocks
        b_blocks = []
        cur_b_c_in = self.c_channels[0]
        for i, (b_ks, b_c_out) in enumerate(zip(self.b_kernel_sizes, self.b_channels)):
            b_blocks.append(QuartzBBlock(cur_b_c_in, b_c_out, n_cells=b_ks))
            b_blocks.extend([QuartzBBlock(b_c_out, b_c_out, n_cells=b_ks)] * (self.n_bblocks - 1))
            cur_b_c_in = b_c_out
        self.b_blocks = Sequential(*b_blocks)

        # Conv-BN-ReLU-2
        self.conv2 = Sequential(
            QuartzBaseModule(self.b_channels[-1], self.c_channels[1], self.c_kernel_sizes[1]),
            self.activation
        )
        # Conv-BN-ReLU-3
        self.conv3 = Sequential(
            QuartzBaseModule(self.c_channels[1], self.c_channels[2], self.c_kernel_sizes[2]),
            self.activation
        )
        # Conv-BN-ReLU-4: pointwise conv, dilation=2
        self.conv4 = nn.Conv1d(self.c_channels[-1], self.n_class, 1, dilation=2)

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram.permute(0, 2, 1)    # ft-first
        x = self.conv1(x)
        x = self.b_blocks(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return {'logits': x.permute(0, 2, 1)}    # time-first

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
