'''
IJCAI2024
our method: adaptive_patch_sampling module

根据切成patches的清晰度值，删减值低40%，保留60%的patches，这些patches携带着自身的原本的为位置信息。
1.切割图像为Patches；
2. 对所有Patches都进行位置编码；
3.计算每个Patch的清晰度；
4.创建清晰度值矩阵：根据计算出的清晰度值，创建一个大小为 (260/13) * (344/8) 的矩阵。
5.根据清晰度值删除Patches：删除清晰度值较低的40%的patches。
6.保留剩余的Patches所对应的位置编码；
7.将剩余的Patches拉平或投影为Tokens：将每个patch拉平或通过一个线性层投影为tokens。
8.组合位置编码和Tokens：将位置编码和tokens结合起来，形成patch embedding。
9. 将Patch Embedding送入Transformer：将得到的patch embedding送入后续的transformer模型。

网络的输出头参考EPC方法中DHGNN网络的输出头形式写的，基于vit的方法的输出头已修改完成并验证可行。
同理，将本方法中的输出头也换成上述方式。
'''


import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
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

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 做了一个映射，将dim=1024 映射到inner_dim * 3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对tensor张量分块x: 1 x 197(196+1cls) X 1024, 1024是每个token embedding的维度，定义为成768，512都可以，在文末中被定义为1024
        # chunk函数将映射成1024*3的维度分成三块，生成的qkv是个元组tuple,长度为3，每个元素的形状为1x197x1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 这里有一个layer normalization,这是在多头注意力和前馈网络之前需要
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
    def __init__(self, *, image_height, image_width, patch_size, num_keypoints, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        patch_height, patch_width = patch_size[0], patch_size[1]
        self.num_joints = num_keypoints
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # patch的个数
        patch_dim = channels * patch_height * patch_width  # 将patch拉平
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 将patch拉平
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),  # 将拉平后的patch映射到encode需要的维数dim, dim在文末定义中为1024
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 生成位置编码，包含CLS符号和所有token对应的位置
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        # # self.mlp_head = nn.Linear(dim, num_classes)
        # self.head = nn.Sequential(
        #     nn.Linear(dim, image_height * image_width),
        #     Rearrange('b (h w) -> b 1 h w', h=image_height, w=image_width),
        #     nn.Conv2d(1, num_keypoints, kernel_size=1),
        #     nn.Upsample(size=(image_height, image_width), mode='bilinear',
        #                 align_corners=False),
        #     nn.Sigmoid()
        # )

        self.linear1 = nn.Linear(dim, 1024, bias=False)  # [1, 1024]
        self.bn6 = nn.BatchNorm1d(1024)  # [4, 1024]
        self.dp1 = nn.Dropout(p=0.1)

        self.linear2 = nn.Linear(1024, 512)  # [4, 1664], 1664=13*128
        self.bn7 = nn.BatchNorm1d(512)

        self.linear3 = nn.Linear(512, mlp_dim)
        self.bn8 = nn.BatchNorm1d(mlp_dim)
        self.dp2 = nn.Dropout(p=0.1)

        self.mlp_head_x = nn.Linear(mlp_dim, self.num_joints * image_width)  # [256, 260]
        self.mlp_head_y = nn.Linear(mlp_dim, self.num_joints * image_height)

    def forward(self, img):
        # 步骤 1: 切割图像为Patches
        x = self.to_patch_embedding(img)  # img，输入（1，1，260，344），输出为（2，860，1024）
        b, n, _ = x.shape  # b is batchsize, n是长度 n=860


        # 步骤 2: 为所有Patches添加位置编码
        # pos_embedding = self.pos_embedding[:, :n]  # 不直接修改 self.pos_embedding torch.Size([1, 860, 1024])
        pos_embedding = self.pos_embedding[:, :n].expand(b, -1, -1)  # 根据批次大小动态调整形状
        x += pos_embedding

        # x += self.pos_embedding[:, :n]  # 拼接后的每个token+各自的position embedding
        x = self.dropout(x)

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 把cls符号复制b份，因为batchsize
        # x = torch.cat((cls_tokens, x), dim=1)  # cls符号和token embedding进行拼接

        # 步骤 3: 对每个Patch计算清晰度
        clarity_values = self.calculate_clarity(x)  # (2, 860)

        # 步骤 4: 创建清晰度矩阵（删除40%的patches）
        keep_percentage = 0.85
        num_patches_to_keep = int(n * keep_percentage)  # 731=860*0.85
        top_indices = clarity_values.argsort()[:, :num_patches_to_keep]  # 在每个批次中找到排名前num_patches_to_keep的索引  torch.Size([2, 731])
        x = torch.stack([x[i, top_indices[i], :] for i in range(b)])  # 保留排名前num_patches_to_keep的元素  [2,731,1024], b=2

        # _, top_indices = clarity_values.flatten().topk(num_patches_to_keep)  #[731]
        # x = x[:, top_indices, :]  # [1,731,1024],b=1

        # 步骤 5: 保留剩余Patches所对应的位置编码
        # self.pos_embedding.data = self.pos_embedding[:, top_indices, :]
        # selected_pos_embedding = pos_embedding[:, top_indices, :]  # [1,731,1024]
        selected_pos_embedding = torch.stack([pos_embedding[i, top_indices[i], :] for i in range(b)])

        # 步骤 6: 拉平或投影剩余的Patches为Tokens
        x = self.to_latent(x)   # [1,731,1024]

        # 步骤 7: 结合位置编码和Tokens
        # x = x + self.pos_embedding
        x = x + selected_pos_embedding  # torch.Size([2, 731, 1024])

        # 步骤 8: 将Patch Embedding送入Transformer
        x = self.transformer(x)  # torch.Size([2, 731, 1024])

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # return self.head(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # torch.Size([2, 1024]), b=2

        x = self.to_latent(x)  # torch.Size([2, 1024])

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)  # torch.Size([1, 1024])
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.linear3(x)
        x = self.dp2(x)

        batch_size = x.size(0)
        preds_x = self.mlp_head_x(x)
        # print('preds_x.shaoe', preds_x.shape)#  (4,3380) 3380=13*260
        preds_y = self.mlp_head_y(x)
        # print('preds_y.shaoe', preds_y.shape)  #
        preds_x = preds_x.view(batch_size, self.num_joints, 320)
        preds_y = preds_y.view(batch_size, self.num_joints, 180)

        return preds_x, preds_y

    @staticmethod
    def calculate_clarity(patches):
        # 这里你可以根据需要定义清晰度的计算方式
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

        return loss / num_joints  # Normalizing by the number of joints


if __name__ == "__main__":
    v = ViTPose(
        image_height=260,
        image_width=344,
        patch_size=(13, 8),  # 共有224 / 16 = 14， 14 * 14 = 96个patch
        num_keypoints=13,
        dim=1024,
        depth=1,  # encode堆叠的个数，就是多少个encode
        heads=16,  # 多头注意分为多少个头，也就是需要几个注意力
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(2, 1, 260, 344)
    img = torch.clamp(img, max=260)
    print('img', img.shape)

    preds_x, preds_y = v(img)
    print('preds', preds_x.shape)

    # targets = torch.randn(1, 13, 260, 344)
    #
    # target_weights = torch.ones(targets.shape[0], targets.shape[1], 1)
    # mse_loss = JointsMSELoss(use_target_weight=True)
    #
    # # Compute the loss
    # loss = mse_loss(preds, targets, target_weights)
    #
    # # Backward pass
    # loss.backward()
    #
    # # Print the loss
    # print(loss.item())
