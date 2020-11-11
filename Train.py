import pandas as pd
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#os.environ["CUDA_VISIBLE_DEVICE"]="4"
device = 'cuda:4'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### parser
parser = argparse.ArgumentParser(description='PyTorch ResNet50 Training')
parser.add_argument('--output_path', default='./model_196/', help='folder to output images and model checkpoints')
args = parser.parse_args()

##### hyperparameter
EPOCH = 100 
pre_epoch = 10
BATCH_SIZE = 128
LR = 0.01
save_freq = 10

df = pd.read_csv('training_labels.csv')
##### how many classes of cars
car_label = df['label']
car_label = car_label.to_numpy()
cars_set = sorted(set(car_label))
cars_list = list(cars_set)
dic = {}
for i in range(len(cars_set)):
    dic[cars_list[i]] = i

##### files id
files_id = df['id'].to_numpy()
image_path = ["training_data/" + str(i).zfill(6) + ".jpg" for i in files_id]
# print(image_path)

##### train_data
if os.path.isfile("number_target.npy"): 
    number_target = np.load("number_target.npy")
    print("number_target: ", number_target)
else:
    number = []
    print("len of car_label : ", len(car_label))
    for i in range(len(car_label)):
        number.append(dic[car_label[i]])
    number_target = np.array(number) 
    np.save("number_target.npy" ,number_target)
    print("number_target: ", number_target)

##### dataloader
normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
                                )
preprocess = transforms.Compose([
             # transforms.Resize(512),
             transforms.RandomCrop(448),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize
                                ])

def default_loader(path):
    img =  Image.open(path).convert('RGB')
    img = img.resize((512,512))
    img_tensor = preprocess(img)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定義好 image 的路徑
        self.images = image_path
        self.target = number_target
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
dataiter = iter(trainloader)
images, labels = dataiter.next()

#model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model = models.resnet50(pretrained=True)
model_in_feature = model.fc.in_features
model.fc = nn.Linear(model_in_feature, 196)
model = model.to(device)
# PATH = "model/save10.pt"
# model.load_state_dict(torch.load(PATH))
# print("loading model from", PATH)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

for epoch in range(pre_epoch+1, EPOCH+1):
    print("starting training")
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.to(device)), Variable(target.to(device, dtype=torch.int64))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(trainloader.dataset),100. * batch_idx / len(trainloader), loss.item()))
    if epoch % save_freq == 0:
        # model.save()
        torch.save(model.state_dict(), args.output_path+'save_'+str(epoch)+'.pt')
