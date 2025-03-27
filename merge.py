import torch
import torch.nn as nn

# 定义基础模型结构
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.head_a1 = nn.Linear(64 * 13 * 13, 10)  # 假设输入尺寸为32x32
        self.head_a2 = nn.Linear(64 * 13 * 13, 5)   # 第二个检测头

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        return self.head_a1(features), self.head_a2(features)

# 初始化合并模型
class MergedModel(BaseModel):
    def __init__(self):
        super().__init__()
        # 此处可扩展其他共享组件

# 权重加载函数
def load_merged_weights(new_model, a1_weights_path, a2_weights_path):
    # 加载原始模型状态字典
    a1_state_dict = torch.load(a1_weights_path)
    a2_state_dict = torch.load(a2_weights_path)
    
    # 创建目标模型的状态字典
    merged_state_dict = new_model.state_dict()
    
    # 合并骨干网络和第一个头的权重 (来自A1)
    for key in a1_state_dict:
        if key.startswith('backbone') or key.startswith('head_a1'):
            merged_state_dict[key] = a1_state_dict[key]
    
    # 合并第二个头的权重 (来自A2)
    for key in a2_state_dict:
        if key.startswith('head_a2'):
            merged_state_dict[key] = a2_state_dict[key]
    
    # 加载合并后的权重
    new_model.load_state_dict(merged_state_dict, strict=True)
    return new_model

# 使用示例
if __name__ == "__main__":
    # 初始化空模型
    merged_model = MergedModel()
    
    # 加载预训练权重
    merged_model = load_merged_weights(
        merged_model,
        a1_weights_path="path/to/a1.pth",
        a2_weights_path="path/to/a2.pth"
    )
    
    # 验证加载结果
    test_input = torch.randn(1, 3, 32, 32)
    out1, out2 = merged_model(test_input)
    print(f"Output shapes: {out1.shape}, {out2.shape}")

