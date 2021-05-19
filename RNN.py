import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import time
from datetime import timedelta
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

#import the csv data 
DATA = pd.read_csv('london_merged.csv')

DATA['timestamp'] = pd.to_datetime(DATA['timestamp'])
DATA['day_week'] = DATA['timestamp'].dt.dayofweek
DATA['day_month'] = DATA['timestamp'].dt.day
DATA['day_year'] = DATA['timestamp'].dt.dayofyear
DATA['hour'] = DATA['timestamp'].dt.hour

DATA = DATA.drop(['t2'], axis = 1)
DATA = DATA.drop(['is_weekend'], axis = 1)
DATA = DATA.drop(['season'], axis = 1)
DATA = DATA.drop(['timestamp'], axis = 1)

#obtain the dimension values to dataset and the variables
print('---------Dimension:', DATA.shape)
print(DATA.columns)
#Any null value?
print('---------Porcentaje de nulos:')
print(DATA.isnull().sum()/ DATA.shape[0])

#Data stadistics
print('---------Data statistics')
print(DATA.describe())

#sns.set()
#sns.heatmap(DATA.corr(), square=True, annot=True, cmap="YlGnBu")

###############################################################################
#      DATA TRATAMENT                                                         #
###############################################################################

DATA['t1']  = DATA['t1'] / (max(DATA['t1'])*1.2)
DATA['t1'] = pd.to_numeric(DATA['t1'], downcast='float')

DATA['hum']  = DATA['hum'] / (max(DATA['hum']*1.1)) * 2 - 1
DATA['hum'] = pd.to_numeric(DATA['hum'], downcast='float')

DATA['wind_speed']  = DATA['wind_speed'] / (max(DATA['wind_speed'])*1.1) * 2 - 1
DATA['wind_speed'] = pd.to_numeric(DATA['wind_speed'], downcast='float')

DATA['hour']  = DATA['hour'] / max(DATA['hour'])
DATA['hour'] = pd.to_numeric(DATA['hour'], downcast='float')

DATA['day_week']  = DATA['day_week'] / max(DATA['day_week'])
DATA['day_week'] = pd.to_numeric(DATA['day_week'], downcast='float')

DATA['day_month']  = DATA['day_month'] / max(DATA['day_month'])
DATA['day_month'] = pd.to_numeric(DATA['day_month'], downcast='float')

DATA['day_year']  = DATA['day_year'] / max(DATA['day_year'])
DATA['day_year'] = pd.to_numeric(DATA['day_year'], downcast='float')

DATA['weather_code']  = DATA['weather_code'] / max(DATA['weather_code'])
DATA['weather_code'] = pd.to_numeric(DATA['weather_code'], downcast='float')

DATA['cnt'] = pd.to_numeric(DATA['cnt'], downcast='float')
DATA['is_holiday'] = pd.to_numeric(DATA['is_holiday'], downcast='float')

data_tr = [] #List with training data
data_val = [] # List with validation data
data_re = [] # List with validation res

datava = np.array(DATA.drop(['cnt'], axis = 1)) #Array data
datare = np.array(DATA['cnt'])  #array with expectec values to predict
for i in range(DATA.shape[0]):
    
    if i < DATA.shape[0]*0.7:
        data_tr.append( (torch.tensor(datava[i]), datare[i]) )
        
    else:
        data_val.append( (torch.tensor(datava[i]), datare[i]) )

dim = len(data_val)
D = len(data_tr)//9

class RNN(nn.Module):
    def __init__(self, num = 9):
        super(RNN, self).__init__()
        self.reg = nn.Sequential(                                                 
            self.linear_block(num, num * 100),
            self.linear_block(num * 100, num * 200, dp = 0.5),
            #self.linear_block(num * 100, num * 20),
            #self.linear_block(num * 20, 3, dp = 0.5),
            self.last_block( num * 200, 1)
        )
        
    def linear_block(self, input, output, dp = 0.0):
        return nn.Sequential(
            nn.Linear(input, output),
            nn.BatchNorm1d(output),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dp),
        )
            
    def last_block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.ReLU()
        )

    def forward(self, data):
        regresion = self.reg(data)
        return regresion

criterion =nn.MSELoss()

#Constant values 
batch_size = 150
#batch_size = len(data_tr)//2
lr = 0.00025 # A learning rate 
beta_1 = 0.5 
beta_2 = 0.999
n_epochs = 5000
display_step = 5

device = 'cuda'
#device = 'cpu'

def weights_init(m):
     torch.nn.init.normal_(m.weight, 0.0, 0.02)

#data_tr = DataLoader(data_tr, batch_size=2000, shuffle=False)
data_tr = DataLoader(data_tr, batch_size=D, shuffle=False)
data_val = DataLoader(data_val, batch_size=dim, shuffle=False)

rnn = RNN().to(device)
rnn_opt = torch.optim.Adam(rnn.parameters(), lr=lr)#, betas=(beta_1, beta_2))

display_step = 100
mean_loss = 0
loss = []
loss_tot = []
step = 0
val_loss = 0

now = time.time()

for epoch in range(n_epochs):
    
    for dat, res in tqdm(data_tr):
        
        #cur_batch_size = len(res)
        dat = dat.to(device)
        pred = rnn(dat)
        res = res.to(device)
        #print(pred.shape)
        #print(RES.shape)
        pred = pred.permute(-1,0)[0]
        pred_loss = criterion(pred, res)
        pred_loss.backward()
        rnn_opt.step()
        
        mean_loss += pred_loss.item() / display_step
        # Keep track of the average generator loss
        
        
        if step % display_step == 0 and step > 0:
            loss.append(mean_loss)
            mean_loss = 0
            step = 0
        step += 1
    
    for dat1, res1 in tqdm(data_val):
    
        dat1 = dat1.to(device)
        pred1= rnn(dat1.detach())
        #pred1= rnn(dat1.detach()).to('cpu')
        res1 = res1.to(device)
        pred1 = pred1.permute(-1,0)[0]
        pred_loss1 = criterion(pred1, res1)
        pr_l = pred_loss1.item()

    #loss_tot.append(pred_loss1)
    loss_tot.append(pr_l)

    if pred_loss1 < val_loss or val_loss == 0:
        torch.save(rnn, 'bestrnn.pth')
        fig1 = plt.figure()
        plt.plot(pred1.cpu().detach().numpy(), res1.cpu().detach().numpy(), label='Prediccion and real')
        #plt.plot(res1.cpu().detach().numpy(), label='Valor real')
        plt.title('Loss:' + str(pred_loss1))
        plt.legend()
        fig1.savefig('Prediction.png')
        plt.close()
        val_loss = pred_loss1
        

    
    timestamp = time.time() - now
    print("Tiempo ejecución:", timedelta(seconds=timestamp), 'Epoca:', epoch, 'de', n_epochs)
    '''
    plt.plot(range(len(loss)),loss, label="MAE Loss")
    plt.legend()
    plt.show()
    '''
    '''
    #print('Funcion de perdida:',pred_loss)
    fig2 = plt.figure() 
    plt.plot(loss_tot, label='loss evolution')  
    plt.legend()
    fig2.savefig('Total_loss.png')
    plt.close() 
      ''' 

timestamp = time.time() - now
print("Tiempo ejecución:", timedelta(seconds=timestamp), 'Epoca:', epoch, 'de', n_epochs)
fig = plt.figure()
plt.plot(range(len(loss)),loss, label="MAE Loss")
plt.legend()
fig.savefig('temp.png')
plt.close()

'''
for dat1, res1 in tqdm(data_val):
    
    dat1 = dat1.to(device)
    pred1= rnn(dat1).to('cpu')
    res1 = res1#.to(device)
    pred1 = pred1.permute(-1,0)[0]
    pred_loss1 = criterion(pred1, res1)
'''



print('Funcion de perdida:',pred_loss)
fig2 = plt.figure() 
plt.plot(loss_tot, label='loss evolution')  
plt.legend()
fig2.savefig('Total_loss.png')
plt.close()     
