#  Homework: Automatic segmentation of Prostate and Prostate peripheral zone (PZ) from MRI images

## Introduction
In this homework we will solve another interesting health related problem using deep learning. 
The objective is to build a model that can automatically segment the prostate and 
the prostate peripheral zone (PZ) from MRI images in 3D.

The dataset was obtained from [MedicalDecathlon](https://medicaldecathlon.com/).

The dataset only contains 20 MRIs of the heart, and 10 images for testing. The objective of the challenge is to segment the left atrium from the MRI images.
The images are in the NIfTI format.

Specific objectives for this homework are to:
1. Apply what we have learned about CNNs to solve a segmentation problem.
2. Continue using TensorBoard to visualize the training process.
3. Obtain experience on solving problems using deep learning from real data in 3D.

## Submission
- `report.md`: A full description of your project. 
- `main.py or main.ipynb`: A python script that will run your project.
- `analizedata.py or ipynb`: A file used to analyze your data
- `mydataset.py`: A custom torch dataset that will load your data.
- `mymodell.py`: A python script that will contain your model or models tested. 
- `training.py`: A python script that will train your model.

### Visualize the data (10 pts)
In your `analizedata` file make a cell or cells that displays four example cases from the training set. 

Each figure should have the following information:
- The T2 image with the segmentation overlayed, displaying the middle slice of the dimension with the 
lowest resolution.  

### Dataset and preprocessing (10 pts)
Create a PyTorch dataset **mydataset.py** that loads the data.

Inside your constructor you should receive at least the *input_folder* and the *transform* to apply to each of the samples.
Please design a method that splits the data into training and validation sets and describe it in your README file.

Please describe your approach for reading the data and preprocessing it. 

How are you predicting the segmentation?

Suggested transformations:
- Resize the images to same isospacing
- Crop the images from the center to a predefined size (e.g. 256x256x256)
- Normalize the images to have zero mean and unit variance (this most be done considering the mean and std of the whole training set)
 
Tip 1: we only have 20 MRI images for training, so you should use a very small validation set.

### Model design (10 pts)
The suggested model to implement is a version of the [3D U-Net](https://arxiv.org/pdf/1606.06650.pdf%E4%BB%A3%E7%A0%81%E5%9C%B0%E5%9D%80%EF%BC%9Ahttps://github.com/wolny/pytorch-3dunet).

But you are free to use any other model that you think is appropriate for this problem.
Please describe with words your proposed model architecture. 
Feel free to include an image of the proposed architecture (from tensorboard or any other tool).

Suggested architecture components:
- 3D convolutional layers
- 3D max pooling layers
- 3D upsampling layers
- 3D batch normalization layers
- 3D residual blocks

### Model training and tensorboard (10 pts)
Create a **train.py** script that will train your model.

The suggested loss function for this segmentation problems is the [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient),
it can be implemented using the [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) library or manually. 

In you training script you should:
- Iterate by a number of epochs and batches, and evaluate the model on the validation set at the end of each epoch.
- Create a tensorboard writer and save the following information for each epoch:
    - Training loss
    - Validation loss
    - A sample of the validation images (2) and the predicted segmentations. 
    - Additionally, save the model graph for the first epoch
- Keep track of the model with the lowest validation loss and save it to disk.

### Model integration (10 pts)
Create a **main.py** file or notebook that will have the general structure of a PyTorch project.
Here you will call your dataset, data loader, model, and training script.

Here you can also include the transformations that you will apply to the data.

### Describe results (10 pts)
Describe the results of your model. Include the following information:
- What is the best validation loss you got?
- Show screenshots of your results from tensorboard. 
- Include a screenshot of the graph and the images. And discuss your results.