import os,glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import mean_squared_error as performance_metric
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.io
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

import models


# Reading the Data. Generating the processed file from original data is at the end of the script
df =  pd.read_csv('./data/housing/processed.csv')#, errors='ignore')
# Shape of the Data
print ('DATA',df.shape)
# df


input_cat_binary_cols = ["renovationCondition","elevator","fiveYearsProperty","subway"] #buildingStructure
input_cat_cols = [] #"buildingStructure"
input_cont_cols = ["square","ladderRatio","Age","floorHeight","livingRoom"]#,,"bathRoom"]
param_cols = ['Lng', 'Lat']
output_cols = ['price']
scale_cols = input_cont_cols+param_cols+output_cols
copy_cols = input_cat_binary_cols+scale_cols
data_processed = df[copy_cols].copy()

for cat_col in input_cat_cols:
    df[cat_col] = df[cat_col].astype(int)
    y = pd.get_dummies(df[cat_col], prefix=cat_col)
    data_processed = pd.concat([data_processed, y], axis=1)


scaler_dict = {}
for col in scale_cols:
    scaler_dict[col] = StandardScaler()
    data_processed[col] = scaler_dict[col].fit_transform(data_processed[[col]])
    
data_processed = shuffle(data_processed,random_state=40)
data_processed = data_processed.rename(dict(zip(list(data_processed.index),range(len(data_processed)))))
# data_processed

param_cols = ['Lng', 'Lat']
output_cols = ['price']
input_cols = list(set(data_processed.keys())-set(param_cols)-set(output_cols))
input_cols.sort()
test_cur = 0
cv_cur = 0

input_dim,param_dim,output_dim = 0,0,0

def get_data(test_cur,val_cur):
    global input_dim,param_dim,output_dim
    kf = KFold(n_splits=5,shuffle=True,random_state=40)
    # kf.get_n_splits(groups_d1r_animal)
    
    train_val_index, test_index= train_test_split(np.arange(len(data_processed)), test_size=0.2, random_state=test_cur)
    data_test = data_processed.iloc[test_index]
    
    for i,(train_index, val_index) in enumerate(kf.split(np.arange(len(train_val_index)))):
        if i!=cv_cur:
            continue
        train_index = train_val_index[train_index]
        val_index = train_val_index[val_index]
        data_train,data_val = data_processed.iloc[train_index],data_processed.iloc[val_index]

    data_train_input = data_train[input_cols].to_numpy()
    data_train_param = data_train[param_cols].to_numpy()
    data_train_output = data_train[output_cols].to_numpy()
    
    data_val_input = data_val[input_cols].to_numpy()
    data_val_param = data_val[param_cols].to_numpy()
    data_val_output = data_val[output_cols].to_numpy()

    data_test_input = data_test[input_cols].to_numpy()
    data_test_param = data_test[param_cols].to_numpy()
    data_test_output = data_test[output_cols].to_numpy()
    # data_test

    input_dim = data_train_input.shape[-1]
    param_dim = data_train_param.shape[-1]
    output_dim = data_train_output.shape[-1]

    trainset = list(zip(data_train_input,data_train_output,data_train_param))
    valset = list(zip(data_val_input,data_val_output,data_val_param))
    testset = list(zip(data_test_input,data_test_output,data_test_param))

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=True,drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True,drop_last=False)
    return train_dataloader,val_dataloader,test_dataloader

train_dataloader,val_dataloader,test_dataloader = get_data(test_cur,cv_cur)
# input_dim,param_dim,output_dim,


# 
metric="r2_score"

if metric=="r2_score":
    from sklearn.metrics import r2_score as performance_metric
    comp_func = np.max
    fill_val = -np.inf
elif metric=="mean_squared_error":
    from sklearn.metrics import mean_squared_error as performance_metric
    comp_func = np.min
    fill_val = np.inf
elif metric=="RMSE":
    from sklearn.metrics import mean_squared_error as mean_squared_error
    def performance_metric(y_true,y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))
    comp_func = np.min
    fill_val = np.inf

ML_model_name = "fc_stg_layered_param_modular_model"#"fc_stg_layered_param_linear_model"#"fc_stg_layered_param_model"#"fc_stg_param_model"

# learning_rate = 1e-2
# stg_regularizer = 1e-2#1e-2 #1e-1
# hidden_dims = [64,128]
# hyper_hidden_dim = [128,64]

epochs = 100
dropout = 0

# for hyper_hidden_dim in [[500],[1000],[64,128],[]]:
#     for stg_regularizer in [1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]:
#         for hidden_dims in [[],[128,64],[1000]]:
#             for learning_rate in [1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]:



gpu = torch.device('cuda:3')

stg = True

for hyper_hidden_dim in [[64,128]]:
    for hidden_dims in [[],[128,64]]:
        for learning_rate in [1e-1,1e-2,1e-3,1e-4]:
            for stg_regularizer in [1e-1,1e-2,1e-3]:
                for rand_init_seed in range(1):
                    for test_cur in range(3):
                        for cv_cur in range(5):
                            print("rand_init_seed:{}, test_cur:{}, cv_cur:{}".format(rand_init_seed,test_cur,cv_cur))

                            train_dataloader,val_dataloader,test_dataloader = get_data(test_cur,cv_cur)

                            # Set the seed value
                        #     seed_value = 42
                            torch.manual_seed(rand_init_seed)
                            np.random.seed(rand_init_seed)

                            root_fname = "/data2/rsristi/FeatureSelection/Trained_Model_housing/corrected_names" #/data2/rsristi/FeatureSelection
                            if not os.path.exists(root_fname):
                                os.mkdir(root_fname)
                                
                            add_name = ""
                            add_name += "_"+"_".join(np.array([input_dim]+hidden_dims+[output_dim]).astype(str))
                            add_name += "_hyper_"+"_".join(np.array([param_dim]+hyper_hidden_dim+[input_dim]).astype(str))

                            model_path = "{}/{}_lr_{}_stg_lr_{}{}_init_{}_test_seed_{}_val_seed_{}_stg_{}.model".format(root_fname,ML_model_name,str(learning_rate).replace(".", "_"),str(stg_regularizer).replace(".","_"),add_name,test_cur,cv_cur,rand_init_seed,stg)

                            loss_path = model_path.replace("model","mat")
                #             plots_folder = model_path.replace(".model","")
                #             if not os.path.exists(plots_folder):
                #                 os.mkdir(plots_folder)
                            if os.path.exists(model_path):
                                continue

                    #         print(stg)
                            model = models.__dict__[ML_model_name](input_dim, hidden_dims, output_dim, param_dim, hyper_hidden_dim, dropout, stg=stg, classification=False)
                            model = model.to(gpu).float()
                    #         print(model)
                            criterion = nn.MSELoss()  # nn.BCELoss() #nn.CrossEntropyLoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))



                            train_metric_array = [fill_val]
                            val_metric_array = [fill_val]
                            test_metric_array = [fill_val]
                            train_loss_array = [0]
                            val_loss_array = [0]
                            test_loss_array = [0]

                            epoch_itr = tqdm(range(epochs), leave=True)
                            for epoch in epoch_itr:

                                train_loss = 0
                                val_loss = 0
                                train_count = 0
                                val_count = 0
                                train_metric_batch = []
                                for batch, (input, target, B) in enumerate(train_dataloader):
                                    model.train()

                                    input = input.to(gpu).float()
                                    target = target.to(gpu).float()
                                    B = B.to(gpu).float()
                            #         B = B.to(gpu).int()/rotations[1]
                            #         B = F.one_hot((B).to(int),num_classes=num_rot).float()
                                    output = model(input,B)
                                    output = torch.squeeze(output)
                                    train_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
                                    loss = criterion(output, torch.squeeze(target)) #.long()
                    #                 print(loss)
                                    if stg:
                                        temp = model.gates.mu
                                        grads = temp.squeeze().cpu().detach().numpy()
                                        stg_loss = torch.mean(torch.abs(model.reg((model.gates.mu + 0.5)/model.sigma))) 
                    #                     print("clf loss:{}, stg_loss:{}".format(loss,stg_loss))
                                        loss += stg_regularizer*stg_loss

                                    optimizer.zero_grad()   
                                    with torch.autograd.set_detect_anomaly(True):
                                        loss.backward()
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                                    optimizer.step()

                                    loss_value = loss.item()
                                    train_loss += loss.item()
                                    train_count += len(input)
                                    #                         train_loss += loss.item()
                #                         train_count += len(input)
                                train_count = batch+1
                                train_metric = np.mean(train_metric_batch)
                                train_metric_array.append(train_metric)
                                train_loss_array.append(train_loss/train_count)

                            #     if (epoch%10==0):#(epoch%5==0) or (epoch==epochs-1):
                                model.eval()

                                val_metric_batch = []
                                val_loss = 0
                                for batch, (input, target, B) in enumerate(val_dataloader):
                                    input = input.to(gpu).float()
                                    target = target.to(gpu).float()
                                    B = B.to(gpu).float()
                                    output = model(input,B)
                                    output = torch.squeeze(output)

                                    val_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
                                    loss = criterion(output, torch.squeeze(target))
                                    if stg:
                                        stg_loss = torch.mean(model.reg((model.gates.mu + 0.5)/model.sigma)) 
                    #                     print("clf loss:{}, stg_loss:{}".format(loss,stg_loss))
                                        loss += stg_regularizer*stg_loss
                            #                 loss += model.l0_reg
                                    val_loss += loss.item()
                                    val_count += len(input)
                                val_count = batch+1
                                val_metric = np.mean(val_metric_batch)
                                val_metric_array.append(val_metric)
                                val_loss_array.append(val_loss/val_count)

                                test_metric_batch = []
                                test_loss = 0
                                test_count = 0
                                for batch, (input, target, B) in enumerate(test_dataloader):
                                    input = input.to(gpu).float()
                                    target = target.to(gpu).float()
                                    B = B.to(gpu).float()
                                    output = model(input,B)
                                    output = torch.squeeze(output)

                                    test_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
                                    loss = criterion(output, torch.squeeze(target))
                                    if stg:
                                        stg_loss = torch.mean(model.reg((model.gates.mu + 0.5)/model.sigma)) 
                    #                     print("clf loss:{}, stg_loss:{}".format(loss,stg_loss))
                                        loss += stg_regularizer*stg_loss
                            #                 loss += model.l0_reg
                                    test_loss += loss.item()
                                    test_count += len(input)
                                test_count = batch+1
                                test_metric = np.mean(test_metric_batch)
                                test_metric_array.append(test_metric)
                                test_loss_array.append(test_loss/test_count)

                                print("epoch {}/{}: cv-{}, train_loss={:.5f}, val_loss={:.5f}, test_loss={:.5f}, train_metric:{:.2f}, val_metric:{:.2f}, test_metric:{:.2f}".format(epoch+1,epochs,cv_cur,train_loss/train_count,val_loss/val_count,test_loss/test_count,train_metric,val_metric,test_metric))

                                if (val_metric == comp_func(val_metric_array)):
                                    torch.save(model.state_dict(), model_path)
                                    print('Model saved! Validation metric improved from {:3f} to {:3f}'.format(comp_func(val_metric_array[:-1]), comp_func(val_metric_array)))
                                scipy.io.savemat(loss_path, {'train_loss_array': train_loss_array,'val_loss_array': val_loss_array,'test_loss_array': test_loss_array,'train_metric': train_metric_array,'val_metric': val_metric_array,'test_metric': test_metric_array})

                #                 if epoch%5==0:
                #                     train_loss = 0
                #                     test_loss = 0
                #                     train_count = 0
                #                     test_count = 0
                #                     train_metric_batch = []
                #                     for batch, (input, target, B) in enumerate(train_dataloader):
                #                         input = input.to(gpu).float()
                #                         target = target.to(gpu).float()
                #                         B = B.to(gpu).float()
                #                 #         B = B.to(gpu).int()/rotations[1]
                #                 #         B = F.one_hot((B).to(int),num_classes=num_rot).float()
                #                         output = model(input,B)
                #                         output = torch.squeeze(output)
                #                         train_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))

                #                         loss = criterion(output, torch.squeeze(target))
                #         #                                 loss = criterion(output, torch.squeeze(target)) #.long()
                #         #                 print(loss)
                #                         if stg:
                #                             temp = model.gates.mu
                #                             grads = temp.squeeze().cpu().detach().numpy()
                #                             stg_loss = torch.mean(torch.abs(model.reg((model.gates.mu + 0.5)/model.sigma))) 
                #         #                     print("clf loss:{}, stg_loss:{}".format(loss,stg_loss))
                #                             loss += stg_regularizer*stg_loss

                #                         train_loss += loss.item()
                #                         train_count += len(input)
                #                     train_count = batch+1
                #                     train_metric = np.mean(train_metric_batch)
                #                     train_metric_array.append(train_metric)
                #                     train_loss_array.append(train_loss/train_count)



                #                     test_metric_batch = []
                #                     test_loss = 0
                #                     for batch, (input, target, B) in enumerate(test_dataloader):
                #                         input = input.to(gpu).float()
                #                         target = target.to(gpu).float()
                #                         B = B.to(gpu).float()
                #                         output = model(input,B)
                #                         output = torch.squeeze(output)

                #                         test_metric_batch.append(performance_metric(target.detach().cpu().numpy(),output.float().detach().cpu().numpy()))
#                         loss = criterion(output, torch.squeeze(target))
#                         if stg:
#                             stg_loss = torch.mean(model.reg((model.gates.mu + 0.5)/model.sigma)) 
#         #                     print("clf loss:{}, stg_loss:{}".format(loss,stg_loss))
#                             loss += stg_regularizer*stg_loss
#                 #                 loss += model.l0_reg
#                         test_loss += loss.item()
#                         test_count += len(input)
#                     test_count = batch+1
#                     test_metric = np.mean(test_metric_batch)
#                     test_metric_array.append(test_metric)
#                     test_loss_array.append(test_loss/test_count)

#     #                 if True:#(epoch%10==0):
#                     print("epoch {}/{}: cv-{}, train_loss={:.5f}, val_loss={:.5f}, test_loss={:.5f}, train_metric:{:.2f}, val_metric:{:.2f}, test_metric:{:.2f}".format(epoch+1,epochs,cv_cur,train_loss/train_count,val_loss/val_count,test_loss/test_count,train_metric,val_metric,test_metric))
#                     if (val_metric == comp_func(val_metric_array)):
#                         torch.save(model.state_dict(), model_path)
#                         print('Model saved! Validation metric improved from {:3f} to {:3f}'.format(comp_func(val_metric_array[:-1]), comp_func(val_metric_array)))
        #             if (test_metric == comp_func(test_metric_array)):
        #                 torch.save(model.state_dict(), model_path)
        #                 print('Model saved! Validation accuracy improved from {:3f} to {:3f}'.format(comp_func(test_metric_array[:-1]), comp_func(test_metric_array)))
#                     scipy.io.savemat(loss_path, {'train_loss_array': train_loss_array,'test_loss_array': test_loss_array,'train_metric': train_metric_array,'test_metric': test_metric_array})

#                     if stg:
#                         B = torch.tensor(data_processed[param_cols].values).to(gpu).float()
#                         mu = model.gates.get_feature_importance(B)
#                         mu = mu.detach().cpu().numpy()
#                         B = B.detach().cpu().numpy()
#                         data_feature_importance = pd.DataFrame(np.hstack((B,mu)),columns=param_cols+input_cols)
#                         for param_col in param_cols:
#                             data_feature_importance[param_col] = scaler_dict[param_col].inverse_transform(data_feature_importance[[param_col]])
#                         plt.figure(figsize=(5*len(input_cols),4))
#                         for col_idx,col in enumerate(input_cols):
#                             plt.subplot(1,len(input_cols),col_idx+1)
#                             fig = plt.scatter(x=data_feature_importance['Lng'], y=data_feature_importance['Lat'], alpha=0.4, \
#                                 c=data_feature_importance[col], cmap=plt.get_cmap('jet'))
#                             plt.colorbar(fig)
#                             plt.title(col)
#                         plt.show()
                        



# # Reading the Data
# df =  pd.read_csv('./data/housing/new.csv',encoding='gbk',low_memory=False)#, errors='ignore')
# # Shape of the Data
# print ('DATA',df.shape)
# df.head(1)
# # Dropping 50% less values feature and no information features
# df.drop(['DOM','url','kitchen','drawingRoom','bathRoom','Cid','id','totalPrice'],axis=1,inplace=True)
# # Dropping missing data rows
# df.dropna(inplace=True)
# df = df[df['constructionTime']!='未知']
# print ('DATA',df.shape)
# # Creating 'distance' feature
# # To calculate Distance Between Two Points on Earth 
# from math import radians, cos, sin, asin, sqrt
# # We will find distance agnaist each lat and lng from Beijing (lat:39.916668,lon:116.383331)
# def distance(lat2, lon2,lat1=39.916668,lon1=116.383331): 
      
#     # The math module contains a function named 
#     # radians which converts from degrees to radians. 
#     lon1 = radians(lon1) 
#     lon2 = radians(lon2) 
#     lat1 = radians(lat1) 
#     lat2 = radians(lat2) 
       
#     # Haversine formula  
#     dlon = lon2 - lon1  
#     dlat = lat2 - lat1 
#     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  
#     c = 2 * asin(sqrt(a))  
     
#     # Radius of earth in kilometers. Use 3956 for miles 
#     r = 6371
       
#     # calculate the result 
#     return(c * r) 
# df['distance'] = df.apply(lambda x: distance (x['Lat'],x['Lng']),axis=1)

# # Creating "Age" feature by substrating from the threshold value "2019"
# df['constructionTime'] = df['constructionTime'].astype(int)
# df['Age'] = 2019 - df['constructionTime']

# # 'timeTrade' feature to year base only.
# df['tradeTime'] = pd.DatetimeIndex(df['tradeTime']).year

# # Creating "floor type" and "floor height" features
# # Fisrt split the string (floor type) and numeric part (the height)
# lst_numeric = []
# lst_str = []
# for value in df['floor'].values:
#     value = value.split()
#     numeric = (value[1])
#     string  = value[0]
#     lst_numeric.append(numeric)
#     lst_str.append(string)

# # Replacing Chinese language words with English words.    
# lst_str_eng=[]
# for string in lst_str:
#     #print(string)
#     if string == '中':
#         lst_str_eng.append(string.replace('中','middle'))
#     elif string == '高':
#         lst_str_eng.append(string.replace('高','high'))
#     elif string == '底':
#         lst_str_eng.append(string.replace('底','bottom'))
#     elif string == '低':
#         lst_str_eng.append(string.replace('低','low'))
#     elif string == '未知':
#         lst_str_eng.append(string.replace('未知','unknown'))
#     elif string == '顶':
#         lst_str_eng.append(string.replace('顶','top'))

# #print (len(lst_str_eng))
# # Converting intto Data Frame or in one shape dataset
# df1 = pd.DataFrame(lst_str_eng,columns=['floorType'])
# df2 = pd.DataFrame(lst_numeric,columns=['floorHeight'])
# df = pd.concat([df,df1,df2],axis=1)
# # Deleting unknown values
# df = df[df['floorType']!='unknown']

# # Dropping missing data which can't be converted into real data
# df.dropna(inplace=True)

# # Dropping features which are not much relevant now.
# df.drop(['floor','constructionTime'],axis=1,inplace=True)

# # Converting 'buildingType' feature to object or string type
# df['buildingType'].replace(1,'Tower',inplace=True)
# df['buildingType'].replace(2,'Bungalow',inplace=True)
# df['buildingType'].replace(3,'Tower and Plate',inplace=True)
# df['buildingType'].replace(4,'Plate',inplace=True)


# # Converting features datatype to see outliers
# df['floorHeight'] = df['floorHeight'].astype(int)
# df['livingRoom'] = df['livingRoom'].astype(int)
# df['district'] = df['district'].astype(int)
# df['tradeTime'] = df['tradeTime'].astype(int)
# df['Age'] = df['Age'].astype(int)
# df['renovationCondition'] = df['renovationCondition'].astype(int)
# df['buildingStructure'] = df['buildingStructure'].astype(int)
# df['elevator'] = df['elevator'].astype(int)
# df['fiveYearsProperty'] = df['fiveYearsProperty'].astype(int)
# df['subway'] = df['subway'].astype(int)
# df['followers']  = df['followers'].astype(int)

# df.livingRoom = df.livingRoom.apply(lambda x: x if x<3 else 3) 
# df.renovationCondition = df.renovationCondition.apply(lambda x: x if x<2 else 3) 
# # Reseting the index
# df.reset_index(inplace=True)
# df.drop(['index'],axis=1,inplace=True)
# # Now the remaining data
# print ("DATA", df.shape)

# # For 'Lng' Outliers
# outliers_Lng = []
# Q1 = 116.344557
# Q3 = 116.481385
# IQR = Q3 - Q1
# for x in df['Lng'].values:
#     if (x < (Q1 - 1.5 * IQR)) or ((Q3 + 1.5 * IQR) < x):
#         if x not in outliers_Lng:
#             outliers_Lng.append(x)

# #print (sorted(outliers_Lng))
# for outlier in outliers_Lng:
#     df = df[df['Lng']!=outlier]

# outliers_Lat = []
# Q1 = 39.894045
# Q3 = 40.012518
# IQR = Q3 - Q1
# for x in df['Lat'].values:
#     if (x < (Q1 - 1.5 * IQR)) or ((Q3 + 1.5 * IQR) < x):
#         if x not in outliers_Lat:
#             outliers_Lat.append(x)

# #print (sorted(outliers_Lat))
# for outlier in outliers_Lat:
#     df = df[df['Lat']!=outlier]

# # For 'distance' Outliers
# outliers_dist = []
# Q1 =  7.821041
# Q3 = 17.444622
# IQR = Q3 - Q1
# for x in df['distance'].values:
#     if (x < (Q1 - 1.5 * IQR)) or ((Q3 + 1.5 * IQR) < x):
#         if x not in outliers_dist:
#             outliers_dist.append(x)

# #print (sorted(outliers_dist))
# for outlier in outliers_dist:
#     df = df[df['distance']!=outlier]

# # For 'Age' Outliers
# outliers_age = []
# Q1 = 13
# Q3 = 25
# IQR = Q3 - Q1
# for x in df['Age'].values:
#     if (x < (Q1 - 1.5 * IQR)) or ((Q3 + 1.5 * IQR) < x):
#         if x not in outliers_age:
#             outliers_age.append(x)

# #print (sorted(outliers_age))
# for outlier in outliers_age:
#     df = df[df['Age']!=outlier]

# # For 'square' Outliers
# outliers_square = []
# Q1 = 58.280000
# Q3 = 99.330000
# IQR = Q3 - Q1
# for x in df['square'].values:
#     if (x < (Q1 - 1.5 * IQR)) or ((Q3 + 1.5 * IQR) < x):
#         if x not in outliers_square:
#             outliers_square.append(x)

# #print (sorted(outliers_square))
# for outlier in outliers_square:
#     df = df[df['square']!=outlier]

# print ('DATA' ,df.shape)
# df.head()

# # df.to_csv("./data/housing/processed.csv",index=False)