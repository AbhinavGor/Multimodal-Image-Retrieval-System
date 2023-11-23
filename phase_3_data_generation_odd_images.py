# Solution for task 2

from PIL import Image
import torchvision
import torch
import cv2
import numpy
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

# Connect to the mongo database
mongo_client = connect_to_mongo()
dbnme = mongo_client.cse515_project_phase1
collection = dbnme.phase3_odd_features

transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

# iterating through all the images in the dataset
for image_ID in range(5000, 8677):
    if image_ID % 2 == 1:
        img, label = dataset[image_ID]

        # resizing the image into 300x10 for Color moment and HoG computation
        resized_img = [cv2.resize(i, (300, 100)) for i in img.numpy()]
        # resizing the image into 224x224 to provide as input to the resnet
        resized_resnet_img = [cv2.resize(i, (224, 224)) for i in img.numpy()]

        # checking if the image has 3 channels
        if len(resized_img) == 3:
            color_moment = extract_color_moment(resized_img)
            hog = extract_hog(resized_img)
            resnet_features = extract_from_resnet(resized_resnet_img)
            collection.insert_one({
                "image_id": str(image_ID),
                "target": label,
                "color_moment": color_moment,
                "hog": hog,
                "avgpool": resnet_features["avgpool"],
                "layer3": resnet_features["layer3"],
                "fc": resnet_features["fc"],
            })
        print(image_ID)
