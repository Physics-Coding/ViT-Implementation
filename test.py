import os
import test
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from utils import create_model
from config import args
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 通常 ViT 输入图片大小为224x224
        transforms.ToTensor(),
    ])

    test_dataset = CIFAR10(root=args.dataset_test_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型
    model = create_model(args)
    # 加载模型权重
    model.load_state_dict(torch.load("summary/vit_base_patch16_224/weights/epoch=100_val_acc=0.7101.pth", map_location=device))
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 预测类别
            output = model(images)
            probabilites = torch.softmax(output, dim=1)
            _, preds = torch.max(probabilites, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # 计算评估指标
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # 准确率
    accuracy = np.mean(true_labels == predicted_labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    # 精确率、召回率和F1-score
    report = classification_report(true_labels, predicted_labels, target_names=test_dataset.classes)
    print("Classification Report:\n", report)

    # 混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", cm)


if __name__ == '__main__':
    main(args)
