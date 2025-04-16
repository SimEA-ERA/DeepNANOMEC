# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:10:46 2023

@author: e.christofi
"""

from tensorflow.keras import models
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 17})

epochs=1500
batch_size = 100
PATH = "../Data"

with open(PATH+'/train_target_1.pkl','rb') as f:
    train_target_1 = pickle.load(f)
    print(train_target_1.shape)


with open(PATH+'/train_input_1.pkl','rb') as f:
    train_input_1 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(train_input_1.shape)     

with open(PATH+'/val_target_1.pkl','rb') as f:
    val_target_1 = pickle.load(f)
    print(val_target_1.shape)


with open(PATH+'/val_input_1.pkl','rb') as f:
    val_input_1 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(val_input_1.shape)    


print("Val_1 e_x:",val_target_1[:,:,:,0].min(),val_target_1[:,:,:,0].max())
print("Val_1 e_yz:",val_target_1[:,:,:,1].min(),val_target_1[:,:,:,1].max())
print("Val_1 s_x:",val_target_1[:,:,:,2].min(),val_target_1[:,:,:,2].max())

print("Train_1 e_x:",train_target_1[:,:,:,0].min(),train_target_1[:,:,:,0].max())
print("Train_1 e_yz:",train_target_1[:,:,:,1].min(),train_target_1[:,:,:,1].max())
print("Train_1 s_x:",train_target_1[:,:,:,2].min(),train_target_1[:,:,:,2].max())
print("")


def normalize(data, col):
    data[:,:,:,col] /= 10000000.
    
    return data


normalize(train_target_1,2)
normalize(val_target_1,2)

print("Val_1 e_x:",val_target_1[:,:,:,0].min(),val_target_1[:,:,:,0].max())
print("Val_1 e_yz:",val_target_1[:,:,:,1].min(),val_target_1[:,:,:,1].max())
print("Val_1 s_x:",val_target_1[:,:,:,2].min(),val_target_1[:,:,:,2].max())

print("Train_1 e_x:",train_target_1[:,:,:,0].min(),train_target_1[:,:,:,0].max())
print("Train_1 e_yz:",train_target_1[:,:,:,1].min(),train_target_1[:,:,:,1].max())
print("Train_1 s_x:",train_target_1[:,:,:,2].min(),train_target_1[:,:,:,2].max())




with open(PATH+'/train_target_2.pkl','rb') as f:
    train_target_2 = pickle.load(f)
    print(train_target_2.shape)


with open(PATH+'/train_input_2.pkl','rb') as f:
    train_input_2 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(train_input_2.shape)     

with open(PATH+'/val_target_2.pkl','rb') as f:
    val_target_2 = pickle.load(f)
    print(val_target_2.shape)


with open(PATH+'/val_input_2.pkl','rb') as f:
    val_input_2 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(val_input_2.shape)    


print("Val_2 e_x:",val_target_2[:,:,:,0].min(),val_target_2[:,:,:,0].max())
print("Val_2 e_yz:",val_target_2[:,:,:,1].min(),val_target_2[:,:,:,1].max())
print("Val_2 s_x:",val_target_2[:,:,:,2].min(),val_target_2[:,:,:,2].max())

print("Train_2 e_x:",train_target_2[:,:,:,0].min(),train_target_2[:,:,:,0].max())
print("Train_2 e_yz:",train_target_2[:,:,:,1].min(),train_target_2[:,:,:,1].max())
print("Train_2 s_x:",train_target_2[:,:,:,2].min(),train_target_2[:,:,:,2].max())
print("")


normalize(train_target_2,2)
normalize(val_target_2,2)


print("Val_2 e_x:",val_target_2[:,:,:,0].min(),val_target_2[:,:,:,0].max())
print("Val_2 e_yz:",val_target_2[:,:,:,1].min(),val_target_2[:,:,:,1].max())
print("Val_2 s_x:",val_target_2[:,:,:,2].min(),val_target_2[:,:,:,2].max())

print("Train_2 e_x:",train_target_2[:,:,:,0].min(),train_target_2[:,:,:,0].max())
print("Train_2 e_yz:",train_target_2[:,:,:,1].min(),train_target_2[:,:,:,1].max())
print("Train_2 s_x:",train_target_2[:,:,:,2].min(),train_target_2[:,:,:,2].max())




with open(PATH+'/train_target_4.pkl','rb') as f:
     train_target_4 = pickle.load(f)
     print(train_target_4.shape)


with open(PATH+'/train_input_4.pkl','rb') as f:
     train_input_4 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
     print(train_input_4.shape)     

with open(PATH+'/val_target_4.pkl','rb') as f:
     val_target_4 = pickle.load(f)
     print(val_target_4.shape)


with open(PATH+'/val_input_4.pkl','rb') as f:
     val_input_4 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
     print(val_input_4.shape)    


print("Val_4 e_x:",val_target_4[:,:,:,0].min(),val_target_4[:,:,:,0].max())
print("Val_4 e_yz:",val_target_4[:,:,:,1].min(),val_target_4[:,:,:,1].max())
print("Val_4 s_x:",val_target_4[:,:,:,2].min(),val_target_4[:,:,:,2].max())

print("Train_4 e_x:",train_target_4[:,:,:,0].min(),train_target_4[:,:,:,0].max())
print("Train_4 e_yz:",train_target_4[:,:,:,1].min(),train_target_4[:,:,:,1].max())
print("Train_4 s_x:",train_target_4[:,:,:,2].min(),train_target_4[:,:,:,2].max())
print("")



normalize(train_target_4,2)
normalize(val_target_4,2)

print("Val_4 e_x:",val_target_4[:,:,:,0].min(),val_target_4[:,:,:,0].max())
print("Val_4 e_yz:",val_target_4[:,:,:,1].min(),val_target_4[:,:,:,1].max())
print("Val_4 s_x:",val_target_4[:,:,:,2].min(),val_target_4[:,:,:,2].max())

print("Train_4 e_x:",train_target_4[:,:,:,0].min(),train_target_4[:,:,:,0].max())
print("Train_4 e_yz:",train_target_4[:,:,:,1].min(),train_target_4[:,:,:,1].max())
print("Train_4 s_x:",train_target_4[:,:,:,2].min(),train_target_4[:,:,:,2].max())



train_input = np.concatenate((train_input_1,train_input_2,train_input_4),axis=0)
train_target = np.concatenate((train_target_1,train_target_2,train_target_4),axis=0)
val_input = np.concatenate((val_input_1,val_input_2,val_input_4),axis=0)
val_target = np.concatenate((val_target_1,val_target_2,val_target_4),axis=0)


train_input = train_input.reshape((-1,100,128,256,9))
train_target = train_target.reshape((-1,100,128,256,3))


indx = [*range(train_input.shape[0])]
np.random.shuffle(indx)
train_input = train_input[indx[:]]
train_target = train_target[indx[:]]

train_input = train_input.reshape((train_input.shape[0]*100,128,256,9))
train_target = train_target.reshape((train_target.shape[0]*100,128,256,3))



def downsample(entered_input,filters, size, apply_batchnorm=True,strides=2):
  
  conv1 = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',use_bias=False)(entered_input) 
  conv1 = tf.keras.layers.LeakyReLU()(conv1)
  
  if apply_batchnorm:
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

  return conv1


def upsample(entered_input,filters, size, skip_layer, apply_dropout=False, strides=2, apply_skip=True):
  tran1 = tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same', use_bias=True)(entered_input)
  tran1 = tf.keras.layers.ReLU()(tran1) 
  if apply_dropout:
      tran1 = tf.keras.layers.Dropout(0.5)(tran1)
  
  if apply_skip:
      tran1 = tf.keras.layers.Concatenate()([tran1,skip_layer])
  return tran1

# Create the Convolutional Neural Network (CNN)
def Generator(): 
  input1 = tf.keras.layers.Input((128,256,9))  
  output1 = downsample(input1, 64, 3)
  output2 = downsample(output1, 128, 3)
  output3 = downsample(output2, 256, 3)  
  output4 = downsample(output3, 512, 3) 
  output5 = downsample(output4, 512, 3) 
  
  output = upsample(output5, 512, 3, output4, apply_dropout=True)
  output = upsample(output, 256, 3, output3, apply_dropout=False)
  output = upsample(output, 128, 3, output2, apply_dropout=False)
  output = upsample(output, 64, 3, output1, apply_dropout=False)
  
  output = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same",  activation="relu")(output)
  out = tf.keras.layers.Conv2DTranspose(3, 3, strides=1, padding="same",  activation=None)(output)

  model = tf.keras.models.Model(input1,out)
  return model



model_u_net= Generator()
model_u_net.summary()

#tf.keras.utils.plot_model(
#     model_u_net, to_file='model.png', show_shapes=True, show_dtype=False,
#     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=300)

MSE=tf.keras.losses.MeanSquaredError()

def compute_region_loss(pred,tar,ind,y_pred):    
    z_pred = tf.zeros_like(y_pred) 
    z_tar = tf.zeros_like(y_pred)
    pred = tf.tensor_scatter_nd_update(z_pred[:,:,:,0],ind,pred)
    non_zero = tf.cast(pred != 0, tf.float32)
   
    pred_mean =tf.math.divide_no_nan( tf.reduce_sum(pred, axis=[1,2]),tf.reduce_sum(non_zero, axis=[1,2]))

    tar = tf.tensor_scatter_nd_update(z_tar[:,:,:,0],ind,tar)
    
    tar_mean =tf.math.divide_no_nan( tf.reduce_sum(tar, axis=[1,2]),tf.reduce_sum(non_zero, axis=[1,2]))
    
    return MSE(pred_mean,tar_mean)





def region_loss(x,y_pred,y_true):
    ind_np = tf.where(tf.equal(x[:,:,:,-5],1))
    ind_intp = tf.where(tf.equal(x[:,:,:,-4],1))
    ind_bulk = tf.where(tf.equal(x[:,:,:,-3],1))
        
    pred_np_ex = tf.gather_nd(y_pred[:,:,:,0],ind_np)
    tar_np_ex = tf.gather_nd(y_true[:,:,:,0],ind_np)
    
    pred_intp_ex = tf.gather_nd(y_pred[:,:,:,0],ind_intp)
    tar_intp_ex = tf.gather_nd(y_true[:,:,:,0],ind_intp)
   
    pred_bulk_ex = tf.gather_nd(y_pred[:,:,:,0],ind_bulk)
    tar_bulk_ex = tf.gather_nd(y_true[:,:,:,0],ind_bulk)
            
    np_ex_loss = compute_region_loss(pred_np_ex, tar_np_ex,ind_np,y_pred)    
    intp_ex_loss = compute_region_loss(pred_intp_ex, tar_intp_ex,ind_intp,y_pred)    
    bulk_ex_loss = compute_region_loss(pred_bulk_ex, tar_bulk_ex,ind_bulk,y_pred)    
    
    
    pred_np_eyz = tf.gather_nd(y_pred[:,:,:,1],ind_np)
    tar_np_eyz = tf.gather_nd(y_true[:,:,:,1],ind_np)
    
    pred_intp_eyz = tf.gather_nd(y_pred[:,:,:,1],ind_intp)
    tar_intp_eyz = tf.gather_nd(y_true[:,:,:,1],ind_intp)
   
    pred_bulk_eyz = tf.gather_nd(y_pred[:,:,:,1],ind_bulk)
    tar_bulk_eyz = tf.gather_nd(y_true[:,:,:,1],ind_bulk)
            
    np_eyz_loss = compute_region_loss(pred_np_eyz, tar_np_eyz, ind_np, y_pred)    
    intp_eyz_loss = compute_region_loss(pred_intp_eyz, tar_intp_eyz, ind_intp, y_pred)    
    bulk_eyz_loss = compute_region_loss(pred_bulk_eyz, tar_bulk_eyz, ind_bulk, y_pred)    
    

    pred_np_sx = tf.gather_nd(y_pred[:,:,:,2],ind_np)
    tar_np_sx = tf.gather_nd(y_true[:,:,:,2],ind_np)
    
    pred_intp_sx = tf.gather_nd(y_pred[:,:,:,2],ind_intp)
    tar_intp_sx = tf.gather_nd(y_true[:,:,:,2],ind_intp)
   
    pred_bulk_sx = tf.gather_nd(y_pred[:,:,:,2],ind_bulk)
    tar_bulk_sx = tf.gather_nd(y_true[:,:,:,2],ind_bulk)
            
    np_sx_loss = compute_region_loss(pred_np_sx, tar_np_sx, ind_np, y_pred)    
    intp_sx_loss = compute_region_loss(pred_intp_sx, tar_intp_sx, ind_intp, y_pred)    
    bulk_sx_loss = compute_region_loss(pred_bulk_sx, tar_bulk_sx, ind_bulk, y_pred)    

    total_loss = np_ex_loss + np_eyz_loss + np_sx_loss + bulk_ex_loss + bulk_eyz_loss + bulk_sx_loss + intp_ex_loss + intp_eyz_loss + intp_sx_loss
    
    return total_loss




def loss_per_atom(x,y_pred,y_true):
    ind_ex = tf.where(tf.not_equal(y_true[:,:,:,0],0))
    pred_ex = tf.gather_nd(y_pred[:,:,:,0],ind_ex)
    tar_ex = tf.gather_nd(y_true[:,:,:,0],ind_ex)

    ind_eyz = tf.where(tf.not_equal(y_true[:,:,:,1],0))
    pred_eyz = tf.gather_nd(y_pred[:,:,:,1],ind_eyz)
    tar_eyz = tf.gather_nd(y_true[:,:,:,1],ind_eyz)
    
    ind_sx = tf.where(tf.not_equal(y_true[:,:,:,2],0))
    pred_sx = tf.gather_nd(y_pred[:,:,:,2],ind_sx)
    tar_sx = tf.gather_nd(y_true[:,:,:,2],ind_sx)

    ex = tf.keras.losses.MeanSquaredError()(pred_ex, tar_ex)    
    eyz = tf.keras.losses.MeanSquaredError()(pred_eyz, tar_eyz)    
    sx = tf.keras.losses.MeanSquaredError()(pred_sx, tar_sx)    

    return ex + eyz + sx





class DNN(tf.keras.Model):
    def __init__(self, NN ,**kwargs):
        super(DNN, self).__init__(**kwargs)
        self.NN = NN()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.region_loss_tracker = tf.keras.metrics.Mean(name="region_loss")
        self.slope_loss_tracker = tf.keras.metrics.Mean(name="slope_loss")
        self.atom_loss_tracker = tf.keras.metrics.Mean(name="atom_loss")
        # self.three_loss_tracker = tf.keras.metrics.Mean(name="three_loss")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.region_loss_tracker,
            self.slope_loss_tracker,
            self.atom_loss_tracker,
            # self.three_loss_tracker
       ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred_x = self.NN(x) 
            
            atom_loss = loss_per_atom(x,pred_x,y)
            r_loss = region_loss(x, pred_x, y)   
            # t_loss = three_phase_loss(x,pred_x,y)
            total_loss = atom_loss + r_loss #+  t_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.atom_loss_tracker.update_state(atom_loss)
        self.region_loss_tracker.update_state(r_loss)
        # self.three_loss_tracker.update_state(t_loss)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "region_loss": self.region_loss_tracker.result(),
            "atom_loss": self.atom_loss_tracker.result(),
            # "three_phase_loss": self.three_loss_tracker.result()
        }

    def test_step(self, input_data):
      x, y = input_data
      pred_x = self.NN(x) 
       
      atom_loss = loss_per_atom(x,pred_x,y)
      r_loss = region_loss(x, pred_x, y)   
      # t_loss = three_phase_loss(x,pred_x,y)
      val_loss = atom_loss + r_loss# + t_loss
             

      return {"loss": val_loss,
              "region_loss": r_loss,
              "atom_loss": atom_loss,
              # "three_phase_loss": t_loss
               } # <-- modify the return value here
  

checkpoint_filepath = './Best_model_weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss', verbose=1,
    mode='min',
    save_best_only=True)

checkpoint_path = "./tmp/model-{epoch:04d}.h5"
#checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path, monitor = 'val_loss',
    verbose=1, save_weights_only=True, mode = "min", 
    save_freq='epoch', save_best_only=False, period=500)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=60, verbose=1, factor=0.8, min_lr=0.000001)
    
model = DNN(Generator)
model.compile(optimizer=tf.keras.optimizers.Adam(),run_eagerly=True)
  
history = model.fit(x=train_input, y=train_target, validation_data=(val_input,val_target), epochs=epochs, batch_size=batch_size, shuffle= False, callbacks=[model_checkpoint_callback,model_checkpoint_callback2,learning_rate_reduction])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("./loss_plot.png")
plt.close()

with open('losses.pkl','wb') as f:
        pickle.dump(history.history, f)
