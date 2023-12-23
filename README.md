# CSE515 - Multimedia and Web Databases

## Project Phase 3
- Project phase 1 and 2 can be accessed <a href="https://github.com/AbhinavGor/Multimodal-Image-Retrieval-System---Phase-1">here</a>.
### System Requirements
1. Python environment (Version 3.10).
2. Windows 10 or MacOS
3. GPU : RTX 3060

### Required libraries:
- matplotlib==3.7.2
- networkx==3.1
- numpy==1.25.2
- opencv-python==4.8.0.76
- pandas==2.1.0
- Pillow==10.0.0
- pymongo==4.5.0
- scikit-learn==1.3.1
- scipy==1.11.2
- tensorly==0.8.1
- torch==2.0.1
- torchvision==0.15.2

### Installation and Execution Instructions

#### Prerequisites:
- Ensure that you have Python 3.x installed on your system.
- Download or clone the project file.
- MongoDB should be installed and running locally.
- Mongo Compass should be installed for easy visual management of your MongoDB databases.
- Download the Caltech 101 dataset and place it in the designated directory within the project.

### The steps to run this project locally are:
1. Open Mongo Compass and connect to your local mongo instance.
2. Go the project directory and run ```pip install -r requirements.txt``` to install all the required python packages for this project.
3. Inherent Dimensionality Computation: Run phase_3_task_0.py to compute the inherent dimensionality of images, once the execution begins the user will be presented with a choice of images and labels, essentially asking to choose between 0a and 0b based on that input we will display the corresponding inherent dimensionality.
4. Execute phase_3_task_1.py along with the feature descriptor that the user wants as one of the arguments. If no latent semantics are found we will be prompted to enter ‘Y’ or ‘N’ for generating the same. Once this is done the script will proceed to make predictions on the odd numbered images and output the respective scoring metrics along with them too.
5. Execute phase_3_task_2.py, this script with process the data from mongodb collection and perform the DBScan clustering along with MDS visualization. The output would be the labels from DBScan clustering and MDS in a 2D output plotter.
6. Execute phase_3_task_3.py. Once the execution begins you will be prompted to select the classifier, feature descriptor. Now based on these choices there would be further selections to be made specific to the classifier:
7. m-NN Classifier: User will be prompted to select the value of m. After the script finishes, it will output the predictions for each test sample. These predictions are the estimated labels based on the m-NN algorithm.
7. Decision-Tree Classifier: There would be no additional inputs and The script returns the predicted class for each test sample, based on the path it takes through the decision tree.
8. PPR Classifier: We will need to additionally give the range of images for PPR, along with the number of connections to be made and the jump probability used for personalization. It also prints actual and predicted labels for each test sample.
9. All of these classifiers will output accuracy, precision, recall, and F1 score for the classification. 
10. Execute phase_3_task_4.py. Once the execution begins the user will be prompted to enter the number of layers and hashes per layer for the LSH when prompted, choose the feature space (e.g., color moment, HoG) you want to use for LSH & input the odd query image ID and the value of k for the top_k similar images. A graphical interface will display images for which you can provide feedback. Click the image and enter your feedback in the text box. Our inputs look like this: {'R+', 'R', 'I', 'I-'}, then submit it. After closing the GUI, the script will save the feedback data to a CSV file named 'task_4_output.csv'.
11. Execute phase_3_task_5.py, Once the execution begins this script will output the SVM's classification of images into different categories based on the feedback. Before passing on the data from the csv file to the SVM, the feedback from the user is assigned the integer values using the following dictionary: {'R+': 0, 'R': 1, 'I': 2, 'I-': 3}. It will also generate a plot or visualization of the classified images. This will take in the .csv file generated from task 4 as the reference.
 
### Contributors-
- Abhinav Gorantla
- Rohan Samuel Gangavarapu
- Ram Abhishek Ramadoss Sivadoss
- Krishna Siddhardh Potluri
- Sensen Wang
