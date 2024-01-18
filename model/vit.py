'''
该代码是基于vit代码实现人体姿态估计。
输入为（260，344，1）的事件帧图像
输出为（batchsize,260,13）和（batchsize,344,13）
，其中13为人体姿态估计的关节点的个数。
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

class HPEViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_joints, dim, depth, heads, mlp_dim, pool = 'cls', channels=1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size[0], image_size[1] # image的大小，一般是(224, 224)
        patch_height, patch_width = patch_size[0], patch_size[1]  # 16 * 16
        self.num_joints = num_joints

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

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 生成位置编码，包含CLS符号和所有token对应的位置
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()  # [1, 1024]

        self.linear1 = nn.Linear(dim, 1024, bias=False)  # [1, 1024]
        self.bn6 = nn.BatchNorm1d(1024)   # [4, 1024]
        self.dp1 = nn.Dropout(p=0.1)

        self.linear2 = nn.Linear(1024, 512)  # [4, 1664], 1664=13*128
        self.bn7 = nn.BatchNorm1d(512)

        self.linear3 = nn.Linear(512, mlp_dim)
        self.bn8 = nn.BatchNorm1d(mlp_dim)
        self.dp2 = nn.Dropout(p=0.1)

        self.mlp_head_x = nn.Linear(mlp_dim, self.num_joints * image_width)  # [256, 260]
        self.mlp_head_y = nn.Linear(mlp_dim, self.num_joints * image_height)


    def forward(self, img):
        x = self.to_patch_embedding(img)  # img，输入（1，3，260，344），输出为（1，860，104）
        b, n, _ = x.shape  # b is batchsize, n是长度 n:860

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 把cls符号复制b份，因为batchsize torch.Size([1, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1)  # cls符号和token embedding进行拼接  torch.Size([1, 861, 768])
        x += self.pos_embedding[:, :(n + 1)]  # 拼接后的每个token+各自的position embedding
        x = self.dropout(x)

        x = self.transformer(x)  # torch.Size([1, 861, 768])

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]   # torch.Size([1, 768])

        x = self.to_latent(x)  # torch.Size([1, 768])

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

if __name__ == "__main__":
    v = HPEViT(
        image_size=(260, 344),
        patch_size=(13, 8),  # 共有260 / 13 = 20， 344 * 8 = 43个patch
        num_joints=13,
        dim=1024,
        depth=6,  # encode堆叠的个数，就是多少个encode
        heads=16,  # 多头注意分为多少个头，也就是需要几个注意力
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(4, 1, 260, 344)

    preds_x, preds_y = v(img)
    print('preds_x.shape', preds_x.shape)   # [4,13,260]
