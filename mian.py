import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        image = Image.open(imagePath).convert("RGB")
        mask = Image.open(maskPath).convert("L")

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        # 将掩码转换为二值化
        mask = (mask > 0).float()

        return image, mask


# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器部分
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 上采样
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)  # 第二次上采样
        self.sigmoid = nn.Sigmoid()  # 输出0到1的值

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)

        x3 = self.upconv1(x2)  # 第一次上采样
        x3 = torch.cat([x3, x1], dim=1)  # skip connection

        x4 = self.conv1(x3)
        x5 = self.upconv2(x4)  # 第二次上采样

        return self.sigmoid(x5)  # Sigmoid激活，输出二值图像


# 数据集路径
imageDatasetPath = "data/imgs"
maskDatasetPath = "data/masks"

# 获取图像和掩码路径
imagePaths = sorted(list(glob.glob(os.path.join(imageDatasetPath, "*.[jp][pn][g]*"))))  # 支持PNG, JPG等格式
maskPaths = sorted(list(glob.glob(os.path.join(maskDatasetPath, "*.[jp][pn][g]*"))))  # 支持PNG, JPG等格式

# 处理GIF掩码文件
gifMaskPaths = sorted(list(glob.glob(os.path.join(maskDatasetPath, "*.gif"))))  # 支持GIF格式
maskPaths.extend(gifMaskPaths)  # 将GIF格式的掩码文件添加到maskPaths中

# 检查图像和掩码路径是否有效
print(f"Number of images: {len(imagePaths)}")
print(f"Number of masks: {len(maskPaths)}")

# 确保图像和掩码路径数量一致
assert len(imagePaths) == len(maskPaths), "Mismatch between number of images and masks!"

# 只取前2000张图像进行训练，剩下的第2001张用于测试
trainImages = imagePaths[:2000]
trainMasks = maskPaths[:2000]

testImages = imagePaths[2000:2001]  # 只取第2001张作为测试图像
testMasks = maskPaths[2000:2001]  # 只取第2001张作为测试掩码

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 创建数据集和数据加载器
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transform)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transform)
trainLoader = DataLoader(trainDS, batch_size=32, shuffle=True)
testLoader = DataLoader(testDS, batch_size=1, shuffle=False)

# 初始化模型、损失函数和优化器
model = UNet().to(device)  # 将模型移至GPU或CPU
criterion = nn.BCEWithLogitsLoss()  # 适合二分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
numEpochs = 50
for epoch in range(numEpochs):
    model.train()
    epoch_loss = 0

    # 使用 tqdm 显示进度条
    with tqdm(trainLoader, desc=f"Epoch {epoch + 1}/{numEpochs}", unit="batch") as tepoch:
        for images, masks in tepoch:
            # 将输入数据移至GPU或CPU
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)  # 使用修改后的输出
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 更新进度条的描述信息
            tepoch.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}/{numEpochs}, Loss: {epoch_loss / len(trainLoader)}")

# 完成训练后，保存最终的完整模型
torch.save(model.state_dict(), "unet_final_complete_model.pth")  # 仅保存模型权重
print("Final complete model saved to 'unet_final_complete_model.pth'")

# 测试模型并输出原图和分割后的图像
def evaluate_and_display(model, testLoader):
    model.eval()
    with torch.no_grad():
        for images, masks in testLoader:
            # 在GPU或CPU上进行推理
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs > 0.5  # 二值化

            # 获取原图和预测图像
            image = images[0].cpu().numpy().transpose(1, 2, 0)  # 转换为HWC格式
            pred_mask = preds[0].cpu().numpy()  # 获取预测掩码
            true_mask = masks[0].cpu().numpy()  # 获取真实掩码

            # 显示原图和分割结果
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')

            plt.show()

# 在训练完成后进行评估
evaluate_and_display(model, testLoader)
