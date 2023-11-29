from PIL import Image
import torchvision
import torch
import cv2
import numpy
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from database_connection import connect_to_mongo

# Connect to the mongo database
mongo_client = connect_to_mongo()
db = mongo_client.cse515_project_phase1
collection = db.phase2_features
image_collection = db.even_images

transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

# iterating through all the images in the dataset
for image_ID in range(8677):
    if image_ID % 2 == 0:
        img, label = dataset[image_ID]
        print(image_ID)
        image_collection.insert_one({"image_ID": str(image_ID), "image": img.numpy().tolist()})
