import os,glob
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

import models

# Define rotations to rotate image 
rotations = [0, 45, 90, 135, 180, 225, 270, 315]
# rotations = [0,90]
num_rot = len(rotations)
digits = [4,9]
# Define transformation to convert image to tensor
transform = transforms.ToTensor()

# Load MNIST train and test datasets
trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
testset = datasets.MNIST('data', train=False, download=True, transform=transform)

# Define filter function to only include images of digits 5 and 6
def filter_dataset(dataset):
    filtered_data = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        if label == digits[0] or label == digits[1]:
            label = 0 if label==digits[0] else 1
            for angle in rotations:
                rotated_img = TF.rotate(image, angle)#torch.rot90(image, angle // 45)
                filtered_data.append((rotated_img.flatten(), label, angle))
                
    return filtered_data

# Filter train and test datasets to only include images of digits 5 and 6
trainset_filtered = filter_dataset(trainset)
testset_filtered = filter_dataset(testset)

# Define dataloaders for train and test datasets
train_dataloader = torch.utils.data.DataLoader(trainset_filtered, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset_filtered, batch_size=32, shuffle=True)

#data dimensions
input_dim = trainset_filtered[0][0].shape[0]
output_dim = 1 



ML_model_name = "fc_stg_layered_param_modular_model"

epochs = 10
param_dim = num_rot
dropout = 0

for hyper_hidden_dim in [[64,128],[1000],[32,64,128]]:
    for hidden_dims in [[],[128,64]]:
        for learning_rate in [1e-3,5e-4,1e-4]:
            for stg_regularizer in [5e-1,1e-1,5e-2,1e-2]:

                add_name = ""
                add_name += "_"+"_".join(np.array([input_dim]+hidden_dims+[output_dim]).astype(str))
                add_name += "_hyper_"+"_".join(np.array([param_dim]+hyper_hidden_dim+[input_dim]).astype(str))
                
                root_fname = "/data2/rsristi/FeatureSelection/Trained_Model_mnist/Trained_Model_mnist" #"./Trained_Model_mnist" #/data2/rsristi/FeatureSelection
                if not os.path.exists(root_fname):
                    os.mkdir(root_fname)
                    
                model_path = "{}/{}_{}_{}_lr_{}_stg_lr_{}{}.model".format(root_fname,ML_model_name,digits[0],digits[1],str(learning_rate).replace(".", "_"),str(stg_regularizer).replace(".","_"),add_name)

                loss_path = model_path.replace("model","mat")
                plots_folder = model_path.replace(".model","")
                if not os.path.exists(plots_folder):
                    os.mkdir(plots_folder)
                
                print(model_path.replace(root_fname,""))
                if os.path.exists(model_path):
                    print("Already Exists")
                    continue
                    
                gpu = torch.device('cuda:3')
                # Load model architecture
                model = models.__dict__[ML_model_name](input_dim, hidden_dims, output_dim, param_dim, hyper_hidden_dim, dropout)
                model = model.to(gpu).float()
                print(model)
                criterion = nn.BCELoss()  
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

                test_acc_array = [0]
                train_acc_array = [0]
                train_loss_array = [0]
                test_loss_array = [0]

                data = tqdm(range(epochs), leave=True)
                for epoch in data:

                    train_loss = 0
                    test_loss = 0
                    train_count = 0
                    test_count = 0
                    
                    for batch, (input, target, B) in enumerate(train_dataloader):
                        model.train()

                        input = input.to(gpu).float()
                        target = target.to(gpu).float()
                        B = B.to(gpu).int()/rotations[1]
                        B = F.one_hot((B).to(int),num_classes=num_rot).float()
                        output = model(input,B)
                        output = torch.squeeze(output)
                        loss = criterion(output, torch.squeeze(target)) #.long()

                        temp = model.gates.mu
                        grads = temp.squeeze().cpu().detach().numpy()

                        stg_loss = torch.mean(torch.abs(model.reg((model.gates.mu + 0.5)/model.sigma))) 
                        loss += stg_regularizer*stg_loss

                        optimizer.zero_grad()   
                        with torch.autograd.set_detect_anomaly(True):
                            loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        optimizer.step()

                        loss_value = loss.item()
                        train_loss += loss.item()
                        train_count += len(input)

                #     if True:#(epoch%5==0) or (epoch==epochs-1):
                    model.eval()

                    train_acc_batch = []
                    for batch, (input, target, B) in enumerate(train_dataloader):
                        input = input.to(gpu).float()
                        target = target.to(gpu).float()
                #             B = B.to(gpu).float()[:,None]/360
                        B = B.to(gpu).int()/rotations[1]
                        B = F.one_hot((B).to(int),num_classes=num_rot).float()
                        output = model(input,B)
                        output = torch.squeeze(output)
                        loss = criterion(output, torch.squeeze(target))
                        train_acc_batch.append(accuracy_score((output>0.5).float().detach().cpu().numpy(), target.detach().cpu().numpy()))
                #                 weights = model.lstm.weight_ih_l0
                #                 l1_loss = torch.sum(torch.abs(weights))
                #                 loss +=  l1_regularizer*l1_loss
                        loss += stg_regularizer*torch.mean(model.reg((model.gates.mu + 0.5)/model.sigma)) 
                #                 loss += model.l0_reg
                        train_loss += loss.item()
                        train_count += len(input)
                    train_acc = np.mean(train_acc_batch)*100
                    train_acc_array.append(train_acc)
                    train_loss_array.append(train_loss/train_count)

                    test_acc_batch = []
                    for batch, (input, target, B) in enumerate(test_dataloader):
                        input = input.to(gpu).float()
                        target = target.to(gpu).float()
                #             B = B.to(gpu).float()[:,None]/360
                        B = B.to(gpu).int()/rotations[1]
                        B = F.one_hot((B).to(int),num_classes=num_rot).float()
                        output = model(input,B)
                        output = torch.squeeze(output)

                        test_acc_batch.append(accuracy_score((output>0.5).float().detach().cpu().numpy(), target.detach().cpu().numpy()))
                        loss = criterion(output, torch.squeeze(target))
                #                 weights = model.lstm.weight_ih_l0
                #                 l1_loss = torch.sum(torch.abs(weights))
                #                 loss +=  l1_regularizer*l1_loss
                        loss += stg_regularizer*torch.mean(model.reg((model.gates.mu + 0.5)/model.sigma)) 
                #                 loss += model.l0_reg
                        test_loss += loss.item()
                        test_count += len(input)
                    test_acc = np.mean(test_acc_batch)*100
                    test_acc_array.append(test_acc)
                    test_loss_array.append(test_loss/test_count)

                    print("epoch {}/{}: train_loss={:.5f}, test_loss={:.5f}, train_acc:{:.2f}, test_acc:{:.2f}".format(epoch+1,epochs,train_loss/train_count,test_loss/test_count,train_acc,test_acc))

                    if (test_acc == np.max(test_acc_array)):
                        torch.save(model.state_dict(), model_path)
                        print('Model saved! Validation accuracy improved from {:3f} to {:3f}'.format(np.max(test_acc_array[:-1]), np.max(test_acc_array)))
                    scipy.io.savemat(loss_path, {'train_loss_array': train_loss_array,'test_loss_array': test_loss_array,'train_acc': train_acc_array,'test_acc': test_acc_array})

#                     B = torch.tensor(rotations).to(gpu).int()/rotations[1]
#                     B = F.one_hot((B).to(int),num_classes=num_rot).float()

#                     mu = model.gates.get_feature_importance(B)
#                     mu = mu.detach().cpu().numpy()

#                     plt.figure(figsize=(5*num_rot,5))
#                     for rotation_idx in range(len(rotations)):
#                         plt.subplot(1,num_rot,rotation_idx+1)
#                         plt.imshow(mu[rotation_idx].reshape(28,28))
#                         plt.title("rotation:{}".format(rotations[rotation_idx]),fontsize=35)
#                         plt.xticks([])
#                         plt.yticks([])
#                     plt.suptitle("epoch:{}, train_acc:{:.2f}, test_acc:{:.2f}".format(epoch,train_acc,test_acc),fontsize=35)
#                     plt.savefig(plots_folder+"/epoch_{}.png".format(epoch))
#                 #     plt.show()







