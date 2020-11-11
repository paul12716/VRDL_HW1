import os
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms.functional as tv_F
from torch.autograd import Variable
import csv
import torchvision.models as models
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:7'

##### make inv_dic
df = pd.read_csv('training_labels.csv')
car_label = df['label'].to_numpy()
cars_set = sorted(set(car_label))
cars_list = list(cars_set)
dic = {}
for i in range(len(cars_set)):
    dic[cars_list[i]] = i
inv_dic = dict(zip(dic.values(), dic.keys()))


##### import model
model = models.resnet50(pretrained=True)
model_in_feature = model.fc.in_features
model.fc = nn.Linear(model_in_feature, 196)
model.load_state_dict(torch.load('model_196/save_100.pt'))
model = model.to(device)
model.eval()

##### import img_list
img_list = os.listdir('testing_data')

##### testing 
with open('Test.csv', 'w', newline='') as csvFile:
    for img_name in img_list:
        print('Processing image: ' + img_name)
        img_id = img_name[0:6]
        img = Image.open(os.path.join('testing_data', img_name)).convert('RGB')
        img = img.resize((512,512))
        img = tv_F.to_tensor(tv_F.resize(img, (512, 512)))
        img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_input = Variable(torch.unsqueeze(img, 0))
        img_input = img_input.to(device)
        output = model(img_input).squeeze()
        print("output result !!! : ", output.shape)
        _, indices = torch.sort(output, descending=True)
        probs = F.softmax(output, dim=-1)
        
        predict = inv_dic[int(indices[0])]

        writer = csv.writer(csvFile)
        writer.writerow([img_id, predict])

