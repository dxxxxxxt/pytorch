import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# 设备检测 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试数据 
test_dataset = torchvision.datasets.ImageFolder(
    root='D:\\pytorch\\test',
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"测试集样本数: {len(test_dataset)}")
print(f"类别: {test_dataset.classes}")

# 加载模型 
# 重新定义模型结构（与训练时完全一致）
num_classes = len(test_dataset.classes)
model = models.resnet18(pretrained=False)

# 冻结所有参数（与训练时一致）
for param in model.parameters():
    param.requires_grad = False

# 修改全连接层
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, num_classes)
)

# 加载训练好的权重
state_dict = torch.load('pokemon_model_best.pth', map_location=device)
model.load_state_dict(state_dict)

# 移动到设备并设置为评估模式
model = model.to(device)
model.eval()

print("模型加载成功！")

# 推理和评估 
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算量化指标
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n========== 模型评估结果 ==========")
print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 分类报告
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# 混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(all_labels, all_preds))