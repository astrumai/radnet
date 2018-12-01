import torch
from PIL import Image
import tifffile as tiff
from torchvision.transforms import Compose, Resize, ToTensor
from preprocessing.load import SegmentationLoader
from torch.utils.data import DataLoader

data_path = "C:\\Users\\Mukesh\\Segmentation\\UnetWork.git\\data\\"

trainPath = data_path + 'train-volume.tif'
labelsPath = data_path + 'train-labels.tif'

# print("shape:", (tiff.imread(trainPath)).shape)
# print("Pil image 1: ", Image.fromarray((tiff.imread(trainPath))[1]))
# print("Pil image 2: ", Image.fromarray((tiff.imread(trainPath))[2]))
# print("Pil image 29: ", Image.fromarray((tiff.imread(trainPath))[29]))

transform = Compose([Resize(64), ToTensor()])
temp = [transform(Image.fromarray((tiff.imread(trainPath))[1]))]
# print("Transform for image 1", temp)

transform = Compose([Resize(64), ToTensor()])
dataset = SegmentationLoader(trainPath, labelsPath, transform)

# print(tiff.imread(trainPath).shape[0])
# print("length of class", (dataset.__getitem__(1)[0][0][0][0:10]))
# print("length of class", (temp[0][0][0][0:10]))

# for i in range(len(dataset)):
#     print(dataset[i])
train_loader = DataLoader(dataset, 4)
print("train_images; len: {}, type: {}, shape {}".format(
    len(train_loader),
    type(train_loader),
    train_loader))

print("Modified file to check git commit")


