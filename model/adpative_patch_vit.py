import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 多头注意力部分
                FeedForward(dim, mlp_dim, dropout=dropout)  # 前馈神经网络
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTPose(nn.Module):
    def __init__(self, *, image_height, image_width, patch_size, num_keypoints, dim, depth, heads, mlp_dim, pool='cls',
                 channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        patch_height, patch_width = patch_size[0], patch_size[1]
        self.num_joints = num_keypoints
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.linear1 = nn.Linear(dim, 1024, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.1)

        self.linear2 = nn.Linear(1024, 512)
        self.bn7 = nn.BatchNorm1d(512)

        self.linear3 = nn.Linear(512, mlp_dim)
        self.bn8 = nn.BatchNorm1d(mlp_dim)
        self.dp2 = nn.Dropout(p=0.1)

        self.mlp_head_x = nn.Linear(mlp_dim, self.num_joints * image_width)
        self.mlp_head_y = nn.Linear(mlp_dim, self.num_joints * image_height)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        pos_embedding = self.pos_embedding[:, :n].expand(b, -1, -1)
        x += pos_embedding

        x = self.dropout(x)

        clarity_values = self.calculate_clarity(x)

        keep_percentage = 0.85
        num_patches_to_keep = int(n * keep_percentage)
        top_indices = clarity_values.argsort()[:, :num_patches_to_keep]
        x = torch.stack([x[i, top_indices[i], :] for i in range(b)])

        selected_pos_embedding = torch.stack([pos_embedding[i, top_indices[i], :] for i in range(b)])

        x = self.to_latent(x)

        x = x + selected_pos_embedding

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # torch.Size([2, 1024]), b=2

        x = self.to_latent(x)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.linear3(x)
        x = self.dp2(x)

        batch_size = x.size(0)
        preds_x = self.mlp_head_x(x)
        preds_y = self.mlp_head_y(x)

        preds_x = preds_x.view(batch_size, self.num_joints, 320)
        preds_y = preds_y.view(batch_size, self.num_joints, 180)

        return preds_x, preds_y

    @staticmethod
    def calculate_clarity(patches):
        clarity = patches.var(dim=-1)
        return clarity


# JointsMSELoss as used in MMPose for 2D Human Pose Estimation
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        """
        Mean squared error loss for joints' heatmaps.

        Parameters:
            use_target_weight (bool): Use target weighting to mask out joints not in the dataset.
        """
        super(JointsMSELoss, self).__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        """
        Forward pass for the loss calculation.

        Parameters:
            output (torch.Tensor): Predicted heatmaps.
            target (torch.Tensor): Ground truth heatmaps.
            target_weight (torch.Tensor): A tensor indicating the weight of each joint target.

        Returns:
            torch.Tensor: The computed mean squared error loss.
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * torch.mean(
                    target_weight[:, idx] *
                    (heatmap_pred - heatmap_gt) ** 2
                )
            else:
                loss += 0.5 * torch.mean(
                    (heatmap_pred - heatmap_gt) ** 2
                )

        return loss / num_joints


if __name__ == "__main__":
    v = ViTPose(
        image_height=260,
        image_width=344,
        patch_size=(13, 8),
        num_keypoints=17,
        dim=1024,
        depth=1,
        heads=16,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(2, 1, 260, 344)
    img = torch.clamp(img, max=260)
    print('img', img.shape)

    preds_x, preds_y = v(img)
    print('preds', preds_x.shape)

