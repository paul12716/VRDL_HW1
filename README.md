# VRDL_HW1

code for Selected Topics in Visual Recognition using Deep Learning Homework 1

## Hardware

- Ubuntu 16.04 LTS
- NVIDIA 1080ti

## Reproducing Submission

1. Download training and testing date
2. Train model by running Train.py
3. Test accuracy by running Test.py and uploading to kaggle competition

competition link : https://www.kaggle.com/c/cs-t0828-2020-hw1/leaderboard

## Download training and testing date

data line : https://drive.google.com/drive/folders/19uUnd4669ljpPreAQx3e7KiD5LY8LlrP

download training_data, testing_data and training_labels.csv

## Train model by running Train.py

In Train.py, we preprocess the data and train our model.

### Data preprocess

We first read training_labels.csv to make a dictionary that correspond a car's class name to a simple int.  
For example : Ford F-150 Regular Cab 2007 correspond to 108.  

Then we do image preprocessing in the following order.
- Resize to 512*512
- Random crop 448*448
- RandomHorizontalFlip
- Normalize

After preprocessing, we use dataloader with batch_size=128 to feed training_data into our model.

### Train our model

```bach
python Train.py
```

We use resnet-50 as backbone network, and modify the last fully connected layer's output dimension to 196.  
196 is the total class numbers of our training_data.  
```python
model = models.resnet50(pretrained=True)
model_in_feature = model.fc.in_features
model.fc = nn.Linear(model_in_feature, 196)
```

We use CrossEntropy as loss function and SGD optimizer, and save model parameters every 10 epochs in total 100 epochs.  
Set learning rate=0.01.  
The model parameters would be saved in folder model_196/

## Test accuracy by running Test.py and uploading to kaggle competition

```bach
python Test.py
```

For testing data, we first make a inverse dictionary that correspond a int to a car's class name.  
For example : 108 correspond to Ford F-150 Regular Cab 2007.  
Then we import the saved model.
```python
model = models.resnet50(pretrained=True)
model_in_feature = model.fc.in_features
model.fc = nn.Linear(model_in_feature, 196)
model.load_state_dict(torch.load('model_196/save_100.pt'))
```

For each testing image, we resize it to 512 * 512, CenterCrop it by 448 * 448 and normalize it.  
In the end, write image id and predicted result to Test.csv.  
Upload it to kaggle competition.  
Best accuracy : 0.92880.
