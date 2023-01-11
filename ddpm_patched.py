
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNetPatched(nn.Module):
    def __init__(
        self, 
        img_shape=(3, 256, 256),
        time_dim=256, 
        hidden=64, 
        level_mult = [1,2,4,8],
        num_middle_layers = 3,
        num_patches=4,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.level_mult = level_mult
        self.num_patches = num_patches

        self.num_levels = len(self.level_mult) - 1

        
        self.channels, self.height, self.width = img_shape
        assert self.height % num_patches == 0, "height of input images needs to be divisible by number of patches"
        assert self.width % num_patches == 0, "width of input images needs to be divisible by number of patches"
        assert self.height == self.width, "only allow square images for now"
        self.img_size = self.height

        # window size when combining patches (after conv layers)
        self.cp_window_size = (num_patches, num_patches)
        self.cp_stride = self.cp_window_size

        # after patch extraction and reshaping the dimension of the first Convs input channel will be:
        # channels * num_patches * num_patches
        self.c_in = self.channels * num_patches * num_patches
        self.c_out = self.c_in

        self.in_layer = DoubleConv(self.c_in, hidden * level_mult[0])

        level = 0
        self.down_conv_layers = []
        self.down_att_layers = []
        for i in range(self.num_levels):
            level += 1
            hidden_in = hidden * level_mult[i]
            hidden_out= hidden * level_mult[i+1]
            self.down_conv_layers.append(
                Down(hidden_in, hidden_out)
            )
            # self attention outputs shape (B, C, S, S) with C channels and S size
            # channels is the same as the layer before self attention outputs
            # i.e. what is given as the second parameter of Down instance (hidden*2 in this case)
            # size is also the same as the previous layer outputs but we first have to calculate the size
            # it is the input image size / num_patches / 2^x with x being the "level" the layer is at within the UNet
            self.down_att_layers.append(
                SelfAttention(hidden_out, self.img_size // self.num_patches // 2**level)
            )
        self.down_conv_layers = nn.ModuleList(self.down_conv_layers)
        self.down_att_layers = nn.ModuleList(self.down_att_layers)
        
        self.middle_layers = []
        hidden_middle = hidden * level_mult[-1]
        # for _ in range(num_middle_layers):
        #     self.middle_layers.append(DoubleConv(hidden_middle, hidden_middle))
        for _ in range(num_middle_layers):
            self.middle_layers.append(DoubleConv(hidden_middle, hidden_middle))
        # # channel count of last middle layer has to fit the channel count of the skip connection with the lowest level
        # # i.e. hidden * factor of second last level
        # self.middle_layers.append(DoubleConv(hidden_middle, hidden * level_mult[-2]))
        self.middle_layers = nn.ModuleList(self.middle_layers)
        
        reversed_level_mult = list(reversed(level_mult))
        self.up_conv_layers = []
        self.up_att_layers = []
        for i in range(self.num_levels):
            level -= 1
            # hidden in takes in the output of the previous layer 
            # (for the first UP it's the last middle layer, for the other UP layers its the respective previous UP)
            # and hidden in takes in some skip connection
            # (UP on level n takes in the same x as the DOWN on level n)
            hidden_in = hidden * reversed_level_mult[i] + hidden * reversed_level_mult[i+1]
            hidden_out= hidden * reversed_level_mult[i+1]
            self.up_conv_layers.append(
                Up(hidden_in, hidden_out)
            )
            self.up_att_layers.append(
                SelfAttention(hidden_out, self.img_size // self.num_patches // 2**level)
            )
        self.up_conv_layers = nn.ModuleList(self.up_conv_layers)
        self.up_att_layers = nn.ModuleList(self.up_att_layers)

        self.out_layer = DoubleConv(hidden * level_mult[0], self.c_out, hidden)
            
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def to_patches(self, x, patch_size=2):
        """Splits tensor x into patches_size*patches_size patches"""
        p = patch_size
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H, W//p, C*p)
        x = x.permute(0, 2, 1, 3).reshape(B, W//p, H//p, C*p*p)
        return x.permute(0, 3, 2, 1)

    def from_patches(self, x, patch_size=2):
        """Combines x's patches_size*patches_size patches into one"""
        p = patch_size
        B, C, H, W = x.shape
        x = x.permute(0,3,2,1).reshape(B, W, H*p, C//p)
        x = x.permute(0,2,1,3).reshape(B, H*p, W*p, C//(p*p))
        return x.permute(0, 3, 1, 2)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x = self.to_patches(x, patch_size=self.num_patches)
        x = self.in_layer(x)

        # Down
        x_down_list = []
        for i in range(self.num_levels):
            x_down_list.append(x)
            conv = self.down_conv_layers[i]
            att = self.down_att_layers[i]
            x = conv(x, t)
            x = att(x)

        for middle_layer in self.middle_layers:
            x = middle_layer(x)

        for i in range(self.num_levels):
            conv = self.up_conv_layers[i]
            att = self.up_att_layers[i]
            x_skip = x_down_list[ self.num_levels-1 - i ]
            x = conv(x, x_skip, t)
            x = att(x)

        x = self.out_layer(x)

        output = self.from_patches(x, patch_size=self.num_patches)
        return output