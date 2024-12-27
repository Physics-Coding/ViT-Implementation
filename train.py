import os
import math
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from config import args
from utils import remove_dir_and_create_dir, create_model, set_seed

def main(args):

    # 数据预处理(训练集 验证集划分)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 通常 ViT 输入图片大小为224x224
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(root=args.dataset_train_dir, train=True, download=True, transform=transform)
    # 划分训练集和验证集
    # 获取训练集大小
    train_size = len(train_dataset)
    # 根据验证集大小计算训练集大小
    val_size = args.val_size
    train_size = train_size - val_size
    # 随机划分训练集和验证集
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    weights_dir = args.summary_dir + "/weights"
    log_dir = args.summary_dir + "/logs"

    remove_dir_and_create_dir(weights_dir)
    remove_dir_and_create_dir(log_dir)
    writer = SummaryWriter(log_dir)

    set_seed(777)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_num, val_num = len(train_subset), len(val_subset)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model = create_model(args)

    if args.freeze_layers:
        for name, params in model.named_parameters():
            # 除classification_head, representation_layer外，其他权重全部冻结
            if "classification_head" not in name and "representation_layer" not in name:
                params.requires_grad_(False)
            else:
                print("training {}".format(name))

    model.to(device)

    # define loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_acc = 0
        train_loss = []
        train_bar = tqdm(train_loader)
        for data in train_bar:
            train_bar.set_description("epoch {}".format(epoch))
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Get model predictions, calculate loss
            logits = model(images)
            prediction = torch.max(logits, dim=1)[1]

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            train_loss.append(loss.item())
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))

            train_acc += torch.eq(labels, prediction).sum()

            # clear batch variables from memory
            del images, labels

        # validate
        model.eval()
        val_acc = 0
        val_loss = []
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = loss_function(logits, labels)
                prediction = torch.max(logits, dim=1)[1]

                val_loss.append(loss.item())
                val_acc += torch.eq(labels, prediction).sum()

                # clear batch variables from memory
                del images, labels

        val_accurate = val_acc / val_num
        train_accurate = train_acc / train_num
        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), train_accurate, np.mean(val_loss), val_accurate))

        writer.add_scalar("train_loss", np.mean(train_loss), epoch)
        writer.add_scalar("train_acc", train_accurate, epoch)
        writer.add_scalar("val_loss", np.mean(val_loss), epoch)
        writer.add_scalar("val_acc", val_accurate, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), "{}/epoch={}_val_acc={:.4f}.pth".format(weights_dir,
                                                                                   epoch,
                                                                                   val_accurate))
            

if __name__ == '__main__':
    main(args)
