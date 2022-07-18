# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# io relatedsklea
# from skimage.io import imread
import os
# from glob import glob
# from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import torch
# %matplotlib inline

from torch.utils.data import DataLoader, Dataset
# from torchvision.utils import make_grid
# from torchvision.utils import save_image
from PIL import Image
# from skimage import io, transform

from tqdm.notebook import tqdm, trange

import torchvision.transforms as transforms

class retinaDataset(Dataset):

    def __init__(self, df, transforms=None):
        'Initialization'
        self.image_size = 512
        self.transforms = transforms

        self.df = df

        # self.train_df, self.valid_df = load_datasets()
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        img_path = self.df["path"][index]
        
        # print(img_path)

        img = Image.open(img_path)
        
        if(self.transforms):
            img = self.transforms(img)

            for i in range(len(img)):
                # print(img[0])
                mean = torch.mean(img[i])
                std = torch.std(img[i])
                img[i] = (img[i] - mean) / std

        return img, torch.tensor(self.df.iloc[index].level)
    
    # def testLen(self):
    #     return len(self.valid_df)
    
    # def getTest(self, index):
    #     img_path = self.train_df["path"][index]
    #     img = Image.open(img_path)
        
    #     if(self.transforms):
    #         img = self.transforms(img)
    #     return img, torch.tensor(self.valid_df.iloc[index].level)


def load_datasets():

    base_data_dir = os.path.join('data')
    training_data_name = "train_1"
    train_data_dir = os.path.join('data', training_data_name)
    train_label_file = 'trainLabels.csv'

    # Load image mapping
    retina_df = pd.read_csv(os.path.join(base_data_dir, train_label_file))
    # Get patient ID
    retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
    # Get image path
    retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(train_data_dir,'{}.jpeg'.format(x)))
    # See if data exists in training data set
    retina_df['exists'] = retina_df['path'].map(os.path.exists)

    print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

    # Left right eye categorical variable
    # 1 is left eye, 0 is right eye
    retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)

    # # Output variable to categorical
    # retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1+retina_df['level'].max()))
    # Remove NA 

    retina_df.dropna(inplace = True)
    retina_df = retina_df[retina_df['exists']]

    rr_df = retina_df[['PatientId', 'level']].drop_duplicates()



    # Split traing and valid sets
    train_ids, valid_ids = train_test_split(rr_df['PatientId'], 
                                    test_size = 0.25, 
                                    random_state = 2018,
                                    stratify = rr_df['level'])
                                    
    train_df = retina_df[retina_df['PatientId'].isin(train_ids)].reset_index(drop = True)  

    valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)].reset_index(drop = True)  

    # print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

    # balance out training data distribution
    # level_0 = int((raw_train_df.level == 0).sum() / 2)
    # level_1 = int((raw_train_df.level == 1).sum() / 2)
    # level_2 = int((raw_train_df.level == 2).sum() / 2)
    # level_3 = int((raw_train_df.level == 3).sum() / 2)
    # level_4 = int((raw_train_df.level == 4).sum() / 2)
    # min_count = min(level_0, level_1, level_2, level_3, level_4)

    # balance size variance in each class
    # train_df = raw_train_df.groupby(['level', 'eye']).apply(lambda x: x.sample(min_count, replace = True)).reset_index(drop = True)                                                   
    # train_df = raw_train_df.groupby(['level', 'eye']).reset_index(drop = True)                                                   

    return train_df, valid_df

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def load_data_loaders(batch_size=1, num_workers=0):
    IMG_SIZE = 224

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_df, valid_df = load_datasets()
    # valid_df, train_df  = load_datasets()

    train_dataset = retinaDataset(train_df, transforms=train_transforms)
    valid_dataset = retinaDataset(valid_df, transforms=valid_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    classes = (0,1,2,3,4,5)

    return train_dataloader, valid_dataloader