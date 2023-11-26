# Functionality: Plots the images produced in the output as a matplotlib figure
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features
rep_collection = db.phase2_representative_images

dataset = torchvision.datasets.Caltech101(
    '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101', download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

def output_plotter(imageIDs):
    # Create a figure where the images can be plotted
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(4, 5, 1)

    fig.suptitle('Query top 10 outputs after updating order based on relevance feedback', fontsize=16)

    for i in range(min(len(imageIDs), 20)):
        img, label = dataset[imageIDs[i]]
        fig.add_subplot(4, 5, i + 1)
        plt.imshow((numpy.squeeze(img)))
        plt.axis('off')
        plt.title("Result ID: " + str(imageIDs[i]))

    plt.show()