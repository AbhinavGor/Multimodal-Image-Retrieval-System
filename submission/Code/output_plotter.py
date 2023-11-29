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
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)


def output_plotter(imageIDs):
    # Create a figure where the images can be plotted
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(4, 5, 1)
    print(imageIDs)
    fig.suptitle(
        f'Query top {len(imageIDs)} outputs after updating order based on relevance feedback', fontsize=16)

    for i in range(min(len(imageIDs), 20)):
        img, label = dataset[imageIDs[i]]
        fig.add_subplot(4, 5, i + 1)
        plt.imshow((numpy.squeeze(img)))
        plt.axis('off')
        plt.title("Rank " + str(i + 1))

    plt.show()


def output_plotter_task_2(imageIDs, title):
    sublists = [imageIDs[i:i + 400] for i in range(0, len(imageIDs), 400)]
    for list in sublists:
        # Create a figure where the images can be plotted
        fig = plt.figure(figsize=(20, 6))
        fig.add_subplot(20, 20, 1)
        fig.suptitle(title, fontsize=16)
        plt.axis('off')
        for i in range(len(list)):
            print(f"printing image {i}")
            img, label = dataset[int(list[i])]
            fig.add_subplot(20, 20, i + 1)
            plt.imshow((numpy.squeeze(img)))
            plt.axis('off')
        plt.show()
