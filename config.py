import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--val_size', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.01)

parser.add_argument('--dataset_train_dir', type=str,
                    default="./dataset/train/",
                    help='The directory containing the train data.')
parser.add_argument('--dataset_test_dir', type=str,
                    default="./dataset/test/",
                    help="The directory containing the test data.")
parser.add_argument('--summary_dir', type=str, default="./summary/vit_base_patch16_224",
                    help='The directory of saving weights and tensorboard.')


# 是否冻结权重
parser.add_argument('--freeze_layers', type=bool, default=False)

parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                    help='The name of ViT model, Select one to train.')
parser.add_argument('--label_name', type=list, default=[
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
], help='The name of class.')

# 测试模型路径
parser.add_argument('--test_model_path', type=str, default="./summary/vit_base_patch16_224/weights/epoch=100_val_acc=0.7101.pth")

args = parser.parse_args()
