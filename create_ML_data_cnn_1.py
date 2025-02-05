# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:48:32 2023

@author: e.christofi
"""
import numpy as np
import pickle
# import random
#from tqdm import tqdm 
np.random.seed(10)

final_target=[]
final_input=[]

strain_step = 1e-5

def calc_phi_np(a):
    v_np = (4./3.)*np.pi*(19.5)**3
    phi =  v_np/(a**3)
    return phi*100

def calc_phi_int(a):
    v_int = (4./3.)*np.pi*(25.5**3-19.5**3)
    phi =  v_int/(a**3)
    return phi*100

a = 52

def norm_stress(data):
    s_min=np.min(data[:,-4])
#    print(s_min)
    data[:,-4]-=s_min
    s_min=np.min(data[:,-4])
#    print(s_min)
    return

phi_np=calc_phi_np(a)
phi_int=calc_phi_int(a)
for i in range(1,16): 
  PATH = "../Raw_data_1/Results"+str(i)+"/Results"   
  for s in range(100,10100,100):
    file=np.loadtxt(PATH+"/step_"+str(s)+".txt")
    g_strain = s*strain_step
#    norm_stress(file)
    g_a = np.ones((file.shape[0],1))
    g_a *= g_strain
    volume_np = np.ones((file.shape[0],1))*phi_np
    volume_int = np.ones((file.shape[0],1))*phi_int
    volume_box = np.ones((file.shape[0],1))*(a)
    file = np.concatenate((file,g_a),axis=1)
    file = np.concatenate((file,volume_np),axis=1)
    file = np.concatenate((file,volume_int),axis=1)
    file = np.concatenate((file,volume_box),axis=1)
    inp_sample_list = []
    tar_sample_list = []
    inp_array = np.zeros([256,13],dtype=np.float64)
    tar_array = np.zeros([256,3],dtype=np.float64)
    for j in range(file.shape[0]):
        inp_sample_list.append(list(file[j,[0,1,2,3,4,5,9,10,11,12,13,14,15]]))
        tar_sample_list.append(list(file[j,[6,7,8]]))
        
        if len(inp_sample_list)==256 and j<10240:
            inp_sample_list = np.array(inp_sample_list,dtype=np.float64)
            tar_sample_list = np.array(tar_sample_list,dtype=np.float64)
            
            inp_array[:]=inp_sample_list[:]
            tar_array[:]=tar_sample_list[:]
            
            final_input.append(inp_array)
            final_target.append(tar_array)
            
            inp_sample_list = []
            tar_sample_list = []
            inp_array = np.zeros([256,13],dtype=np.float64)
            tar_array = np.zeros([256,3],dtype=np.float64)                
        elif len(inp_sample_list)==29 and j>=10240:
          
            inp_sample_list = np.array(inp_sample_list,dtype=np.float64)
            tar_sample_list = np.array(tar_sample_list,dtype=np.float64)
            
            inp_array[113:113+29]=inp_sample_list[:]
            tar_array[113:113+29]=tar_sample_list[:]
            
            final_input.append(inp_array)
            final_target.append(tar_array)
            
            inp_sample_list = []
            tar_sample_list = []
            inp_array = np.zeros([256,13],dtype=np.float64)
            tar_array = np.zeros([256,3],dtype=np.float64)                


    for j in range(87):
         final_input.append(inp_array)
         final_target.append(tar_array)
               
         
final_target = np.array(final_target,dtype=np.float64)
final_input = np.array(final_input,dtype=np.float64)
input_file = np.reshape(final_input,newshape=(int(final_input.shape[0]/128),128,final_input.shape[1],final_input.shape[2]))
target_file = np.reshape(final_target,newshape=(int(final_target.shape[0]/128),128,final_target.shape[1],final_target.shape[2]))

print(input_file.shape, target_file.shape)  

test_input = input_file[1400:]
test_target = target_file[1400:]


#c = [*range(1350)]
#np.random.shuffle(c)
train_input = input_file[:1300]
train_target = target_file[:1300]


val_input = input_file[1300:1400]
val_target = target_file[1300:1400]


print("Train Input file shape:",train_input.shape,"Train Target file shape:",train_target.shape)
print("Test Input file shape:",test_input.shape,"Test Target file shape:",test_target.shape)
print("Val Input file shape:",val_input.shape,"Val Target file shape:",val_target.shape)




with open('train_target_1.pkl','wb') as f:
      pickle.dump(train_target, f) 

with open('train_input_1.pkl','wb') as f:
      pickle.dump(train_input, f) 


with open('test_target_1.pkl','wb') as f:
      pickle.dump(test_target, f) 

with open('test_input_1.pkl','wb') as f:
      pickle.dump(test_input, f) 


with open('val_target_1.pkl','wb') as f:
      pickle.dump(val_target, f) 

with open('val_input_1.pkl','wb') as f:
      pickle.dump(val_input, f) 







