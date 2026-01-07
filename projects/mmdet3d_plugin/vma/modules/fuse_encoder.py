import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils import Registry, build_from_cfg
from ..builder import FUSE_ENCODERS


class ECA(nn.Module):
    """ECA注意力机制模块，高效捕获通道间依赖关系
               高效的attention                 """
    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        # 根据通道数自适应计算卷积核大小
        t = int(abs((torch.log2(torch.tensor(channel)) + b) / gamma))
        k_size = t if t % 2 else t + 1  # 确保卷积核为奇数
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化获取通道特征
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()  # 生成注意力权重

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        b, c, h, w = x.size()
        
        # 全局平均池化: [b, c, h, w] -> [b, c, 1, 1]
        y = self.avg_pool(x)
        
        # 调整维度适应1D卷积: [b, c, 1, 1] -> [b, 1, c]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 1D卷积捕获通道依赖: [b, 1, c] -> [b, 1, c]
        y = self.conv(y)
        
        # 恢复维度并生成注意力权重: [b, 1, c] -> [b, c, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        # 应用注意力权重: 突出重要通道特征
        return x * y.expand_as(x)


@FUSE_ENCODERS.register_module()
class SimpleFusionEncoder(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, conv_cfg=None, norm_cfg=None):
        super().__init__()
        # 第一层卷积：9通道->64通道，提取初步融合特征
        self.conv1 = ConvModule(
            in_channels,
            64,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        
        # 添加ECA注意力：学习64个通道的重要性权重
        self.eca = ECA(channel=64)
        
        # 第二层卷积：64通道->3通道，输出适配backbone的特征图
        self.conv2 = ConvModule(
            64,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
            )
    
    # def forward(self, img, view_img, z_map_img):
    def forward(self, img, view_img):

        """
        融合三通道图像：lidar_map(3) + view_img(3) + z_map_img(3)
        """
        # 按通道拼接三图，形成9通道输入 [B,9,H,W]
        
        # x = torch.cat([img, view_img, z_map_img], dim=1)
        x = torch.cat([img, view_img], dim=1)

        # 第一层卷积提取特征 [B,7,H,W] -> [B,64,H,W]
        x = self.conv1(x)
        
        # ECA注意力加权：动态调整64个通道的权重 [B,64,H,W] -> [B,64,H,W]
        x = self.eca(x)
        
        # 第二层卷积压缩通道 [B,64,H,W] -> [B,3,H,W]
        x = self.conv2(x)
        
        return x


def build_fuse_encoder(cfg, default_args=None):
    """根据配置构建融合编码器"""
    return build_from_cfg(cfg, FUSE_ENCODERS, default_args)


# 测试代码：验证三图融合+ECA注意力的维度正确性
if __name__ == "__main__":
    # 模拟输入：3通道+3通道+1通道
    img = torch.rand([2, 3, 1000, 1000])       # lidar_map
    view_img = torch.rand([2, 3, 1000, 1000])  # view_img
    z_map_img = torch.rand([2, 1, 1000, 1000]) # z_map_img
    
    # 初始化模型（输入7通道，输出3通道）
    model = SimpleFusionEncoder(in_channels=6, out_channels=3)
    # out = model(img, view_img, z_map_img)
    
    # 前向传播
    out = model(img, view_img)
    
    # 打印维度变化过程
    print(f"输入1 (img): {img.shape}")
    print(f"输入2 (view_img): {view_img.shape}")
    print(f"输入3 (z_map_img): {z_map_img.shape}")
    print(f"拼接后: {torch.cat([img, view_img, z_map_img], dim=1).shape}")  # [2,7,1000,1000]
    print(f"conv1输出: {model.conv1(torch.cat([img, view_img, z_map_img], dim=1)).shape}")  # [2,64,1000,1000]
    print(f"ECA输出: {model.eca(model.conv1(torch.cat([img, view_img, z_map_img], dim=1))).shape}")  # [2,64,1000,1000]
    print(f"最终输出: {out.shape}")  # [2,3,1000,1000]（适配backbone输入）
    