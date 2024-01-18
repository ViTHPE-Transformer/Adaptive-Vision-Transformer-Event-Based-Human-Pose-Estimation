import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# adaptive token sampling functions and classes
def log(t, eps=1e-6):
    return torch.log(t + eps)


class PredNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), dropout=0.)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


# 定义损失函数
class KeypointMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        # 假设 preds 和 target 都是形状 (batch_size, num_keypoints * 2)
        # 你需要确保 target 是相同的形状
        return F.mse_loss(preds, target)


class AdaptiveTokenSampling(nn.Module):
    def __init__(self, output_num_tokens, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens

    def forward(self, attn, value, mask):
        heads, output_num_tokens, eps, device, dtype = attn.shape[
                                                           1], self.output_num_tokens, self.eps, attn.device, attn.dtype

        # 提取CLS令牌对所有其他令牌的注意力值
        cls_attn = attn[..., 0, 1:]  # (1,16,256)

        # calculate the norms of the values, for weighting the scores, as described in the paper
        # 计算值的norm，用于加权注意力分数
        value_norms = value[..., 1:, :].norm(dim=-1)  # (1, 16, 256)

        # value_norms = value_norms.detach().cpu().numpy()
        # 计算信息熵
        value_norms_entropy = -torch.sum(value_norms * torch.log(value_norms + 1e-9), axis=-1, keepdims=True)
        # value_norms_entropy = -np.sum(value_norms * np.log(value_norms + 1e-9), axis=-1, keepdims=True)  # 防止对数为负无穷

        # weigh the attention scores by the norm of the values, sum across all heads
        # 加权注意力分数，跨所有头部求和
        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms_entropy)  # (1, 256)

        # normalize to 1
        normed_cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + eps)  # (1,256)
        # normed_cls_attn.shape (1, 256)
        # instead of using inverse transform sampling, going to invert the softmax and use gumbel-max sampling instead
        # 生成伪logits，用于Gumbel-Max采样
        pseudo_logits = log(normed_cls_attn)  # (1, 256)

        # mask out pseudo logits for gumbel-max sampling
        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)

        # expand k times, k being the adaptive sampling number
        # 应用Gumbel-Max采样来选择令牌
        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k=output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, device=device, dtype=dtype)

        # gumble-max and add one to reserve 0 for padding / mask
        sampled_token_ids = pseudo_logits.argmax(dim=-1) + 1

        # calculate unique using torch.unique and then pad the sequence from the right
        unique_sampled_token_ids_list = [torch.unique(t, sorted=True) for t in torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first=True)

        # calculate the new mask, based on the padding
        new_mask = unique_sampled_token_ids != 0

        # CLS token never gets masked out (gets a value of True)
        new_mask = F.pad(new_mask, (1, 0), value=True)

        # prepend a 0 token id to keep the CLS attention scores
        unique_sampled_token_ids = F.pad(unique_sampled_token_ids, (1, 0), value=0)
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h=heads)

        # gather the new attention scores
        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim=2)
        # new _attn.shape (1,16,97,257), new_mask.shape (1,97) unique_sampled_token_ids.shape(1,97)
        # return the sampled attention scores, new mask (denoting padding), as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., output_num_tokens=None):
        super().__init__()
        inner_dim = dim_head * heads  #
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.output_num_tokens = output_num_tokens
        self.ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # dim=1024
            nn.Dropout(dropout)
        )

    def forward(self, x, *, mask):
        num_tokens = x.shape[1]  # 257
        x = self.norm(x)  # (1,257,1024)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # tuple, 3 elmetns, each elments q.shape(1,257,1024)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)  # q (1,16,257,64), heads=16, dim_head=64, k,v same as q

        # 注意力计算
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (1, 16, 257, 257)

        if exists(mask):
            dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)

        attn = self.attend(dots)  # (1, 16,257,257)
        attn = self.dropout(attn)

        sampled_token_ids = None

        # 应用自适应令牌采样
        # if adaptive token sampling is enabled
        # and number of tokens is greater than the number of output tokens
        if exists(self.output_num_tokens) and (num_tokens - 1) > self.output_num_tokens:
            attn, mask, sampled_token_ids = self.ats(attn, v, mask=mask)
            # 如果设置了 output_num_tokens 且序列长度大于 output_num_tokens，
            # 则会调用 AdaptiveTokenSampling类来执行自适应令牌采样。

        out = torch.matmul(attn, v)  # (1,16,257,64)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (1,257,1024)

        return self.to_out(out), mask, sampled_token_ids


class Transformer(nn.Module):
    def __init__(self, dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        assert len(
            max_tokens_per_depth) == depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(max_tokens_per_depth, reverse=True) == list(
            max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        # self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _, output_num_tokens in zip(range(depth), max_tokens_per_depth):
            self.layers.append(nn.ModuleList([
                # PredNorm(dim, Attention(dim, output_num_tokens=output_num_tokens, heads=heads, dim_head=dim_head, dropout=dropout)),
                # PredNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                Attention(dim, output_num_tokens=output_num_tokens, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)

            ]))

    def forward(self, x):
        b, n, device = *x.shape[:2], x.device  # b=1, n=257

        # use mask to keep track of the paddings when sampling tokens
        # as the duplicates (when sampling) are just removed, as mentioned in the paper
        mask = torch.ones((b, n), device=device, dtype=torch.bool)  # (1, 257)

        token_ids = torch.arange(n, device=device)  # (257,)
        token_ids = repeat(token_ids, 'n -> b n', b=b)  # (1, 257)

        for attn, ff in self.layers:
            attn_out, mask, sampled_token_ids = attn(x, mask=mask)  # attn_out.shape(1,257,1024)

            # when token sampling, one needs to then gather the residual tokens with the sampled token ids
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim=1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim=1)

            x = x + attn_out  # (1,257,1024)

            x = ff(x) + x

        return x, token_ids


class PoseViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_keypoints, dim, depth, max_tokens_per_depth, heads, mlp_dim,
                 pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size[0], image_size[1]
        patch_height, patch_width = patch_size[0], patch_size[1]

        self.num_keypoints = num_keypoints
        self.patch_size = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 修改patch embedding来处理单通道输入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)  # 1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # (1, 257, 1024)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # (1,1,1024)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.linear1 = nn.Linear(dim, 1024, bias=False)  # [1, 1024]
        self.bn6 = nn.BatchNorm1d(1024)  # [4, 1024]
        self.dp1 = nn.Dropout(p=0.1)

        self.linear2 = nn.Linear(1024, 512)  # [4, 1664], 1664=13*128
        self.bn7 = nn.BatchNorm1d(512)

        self.linear3 = nn.Linear(512, mlp_dim)
        self.bn8 = nn.BatchNorm1d(mlp_dim)
        self.dp2 = nn.Dropout(p=0.1)

        self.mlp_head_y = nn.Linear(mlp_dim, self.num_keypoints * image_height)  # [256, 260]
        self.mlp_head_x = nn.Linear(mlp_dim, self.num_keypoints * image_width)

    def forward(self, img, return_sampled_token_ids=False):
        # img = img.unsqueeze(1) # 将（2，260，344）变成（2，1，260，344）
        x = self.to_patch_embedding(img)  # if img.shape=(b,1,260,344),就不加上一句
        b, n, _ = x.shape  # batchsize:1, n:256

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # shape (1,1,1024)
        x = torch.cat((cls_tokens, x),
                      dim=1)  # shape (1, 257, 1024), 257 = 256 +1cls, 256=16*16 represents a patch feature, cause patch_weight=16, patch_height=16

        pos_embedding = self.pos_embedding[:, :(n + 1)].expand(b, -1, -1)
        x += pos_embedding

        # x += self.pos_embedding[:, :(n + 1)]   #  (1, 257, 1024)
        x = self.dropout(x)

        clarity_values = self.calculate_clarity(x)  # (2, 860)

        # 步骤 4: 创建清晰度矩阵（删除40%的patches）
        keep_percentage = 0.85
        num_patches_to_keep = int(n * keep_percentage)  # 731=860*0.85
        top_indices = clarity_values.argsort()[:,
                      :num_patches_to_keep]  # 在每个批次中找到排名前num_patches_to_keep的索引  torch.Size([2, 731])
        x = torch.stack([x[i, top_indices[i], :] for i in range(b)])  # 保留排名前num_patches_to_keep的元素  [2,731,1024], b=2

        selected_pos_embedding = torch.stack([pos_embedding[i, top_indices[i], :] for i in range(b)])

        # 步骤 6: 拉平或投影剩余的Patches为Tokens
        x = self.to_latent(x)  # [1,731,1024]

        # 步骤 7: 结合位置编码和Tokens
        # x = x + self.pos_embedding
        x = x + selected_pos_embedding  # torch.Size([2, 731, 1024])

        x, token_ids = self.transformer(x)
        # print(x.shape)  # torch.Size([1, 7, 1024])
        # print(x[:, 0].shape)  # torch.Size([1, 1024])

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
        preds_x = preds_x.view(batch_size, self.num_keypoints, 320)
        preds_y = preds_y.view(batch_size, self.num_keypoints, 180)

        return preds_x, preds_y

    @staticmethod
    def calculate_clarity(patches):
        # 这里你可以根据需要定义清晰度的计算方式
        clarity = patches.var(dim=-1)
        return clarity


if __name__ == '__main__':
    num_keypoints = 13
    v = PoseViT(
        image_size=(260, 344),
        patch_size=(13, 8),
        num_keypoints=num_keypoints,
        dim=768,
        depth=2,
        max_tokens_per_depth=(860, 731),
        heads=16,
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1
    )

    # # 创建损失函数实例
    # criterion = KeypointMSELoss()
    img = torch.randn(2, 1, 260, 344)
    # target = torch.randn(1, num_keypoints, 260, 344)  # 假设的目标关键点位置

    preds_x, preds_y = v(img)  # (4, 1000)
    print(preds_x)

    # # 计算损失
    # loss = criterion(preds, target)

    # print('Loss:', loss.item())
    print('Predicted keypoint positions shape:', preds_x.shape)
