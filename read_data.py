from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image  
import os

# 定义预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集的函数
def make_dataset(root_dir, transform=None):
    """
    返回一个包含所有类别图片的 Dataset
    """
    all_images = []
    labels = []
    
    # 获取所有类别文件夹
    classes = os.listdir(root_dir)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    print(f"找到类别: {classes}")
    
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        count = 0
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            all_images.append(img_path)
            labels.append(class_to_idx[class_name])
            count += 1
        print(f"类别 '{class_name}' 有 {count} 张图片")
    
    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
    
    return CustomDataset(all_images, labels, transform)

# 使用
if __name__ == "__main__":
    print("开始加载数据集...")
    train_dataset = make_dataset('D:\\深度学习\\train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"\n训练集样本总数: {len(train_dataset)}")
    print(f"类别数: {len(set(train_dataset.labels))}")
    
    # 测试取一个 batch
    for images, labels in train_loader:
        print(f"一个 batch 的图像形状: {images.shape}")
        print(f"一个 batch 的标签: {labels[:5]}")
        break