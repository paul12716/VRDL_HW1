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


use simple resnet-50
modify the last layer to 196 output dimensions

run Train.py and save model in model_196/

run Test.py and get Test.csv as predicted result

Best accuracy : 0.92880
