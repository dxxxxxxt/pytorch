import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

writer = SummaryWriter('runs/pokemon_experiment')

train_dataset = torchvision.datasets.ImageFolder(
    root='D:\\pytorch\\train',
    transform=train_transform
)
test_dataset = torchvision.datasets.ImageFolder(
    root='D:\\pytorch\\test',
    transform=test_transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#模型 
model = models.resnet18(pretrained=True)

# 冻结所有预训练参数
for param in model.parameters():
    param.requires_grad = False

# 修改全连接层（适配宝可梦数据集）
num_classes = len(train_dataset.classes)  # 自动获取类别数
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, num_classes)
)

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 损失函数和优化器 
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.fc.parameters(), lr=0.01, weight_decay=1e-4)

# 训练 
num_epochs = 30
best_acc = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), 'pokemon_model_best.pth')

# 保存最终模型
torch.save(model.state_dict(), 'pokemon_model_final.pth')
print("训练完成，模型已保存")