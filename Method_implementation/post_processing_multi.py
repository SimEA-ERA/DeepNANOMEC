# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:10:46 2023

@author: e.christofi
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
# from collections import defaultdict
# from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
# from scipy.stats import linregress
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 18})
import time


epochs=1500
batch_size = 100
PATH = "../Data"





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
  input1 = tf.keras.layers.Input((128,256,10))  
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

def np_loss_ex(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-5],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,0],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,0],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))

def intp_loss_ex(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-4],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,0],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,0],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))

def bulk_loss_ex(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-3],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,0],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,0],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))


def np_loss_eyz(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-5],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,1],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,1],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))

def intp_loss_eyz(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-4],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,1],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,1],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))

def bulk_loss_eyz(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-3],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,1],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,1],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))


def np_loss_sx(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-5],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,2],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,2],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))

def intp_loss_sx(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-4],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,2],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,2],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))

def bulk_loss_sx(x,y_pred,y_true):
    ind = tf.where(tf.equal(x[:,:,-3],1))
    pred_eyz = tf.gather_nd(y_pred[:,:,2],ind)
    tar_eyz = tf.gather_nd(y_true[:,:,2],ind)
            
    pred_mean = tf.math.reduce_mean(pred_eyz)     
    tar_mean = tf.math.reduce_mean(tar_eyz)     
 
    return tf.math.abs(tf.math.subtract(pred_mean,tar_mean))



class DNN(tf.keras.Model):
    def __init__(self, NN ,**kwargs):
        super(DNN, self).__init__(**kwargs)
        self.NN = NN()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.e_x_loss_tracker = tf.keras.metrics.Mean(name="e_x_loss")
        self.e_yz_loss_tracker = tf.keras.metrics.Mean(name="e_yz_loss")
        self.s_x_loss_tracker = tf.keras.metrics.Mean(name="s_x_loss")
        self.e_x_global_loss_tracker = tf.keras.metrics.Mean(name="e_x_global_loss")
        self.e_yz_global_loss_tracker = tf.keras.metrics.Mean(name="e_yz_global_loss")
        self.s_x_global_loss_tracker = tf.keras.metrics.Mean(name="s_x_global_loss")        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.e_x_loss_tracker,
            self.e_yz_loss_tracker,
            self.s_x_loss_tracker,
            self.e_x_global_loss_tracker,
            self.e_yz_global_loss_tracker,
            self.s_x_global_loss_tracker,
       ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred_x = self.NN(x) 
            e_x_loss = tf.keras.losses.MeanSquaredError()(pred_x[:,:,0], y[:,:,0])
            e_yz_loss = tf.keras.losses.MeanSquaredError()(pred_x[:,:,1], y[:,:,1])
            s_x_loss = tf.keras.losses.MeanSquaredError()(pred_x[:,:,2], y[:,:,2])
            m_eyz_np_loss = np_loss_eyz(x,pred_x,y)
            m_eyz_intp_loss = intp_loss_eyz(x,pred_x,y)
            m_eyz_bulk_loss = bulk_loss_eyz(x,pred_x,y)

            m_ex_np_loss = np_loss_ex(x,pred_x,y)
            m_ex_intp_loss = intp_loss_ex(x,pred_x,y)
            m_ex_bulk_loss = bulk_loss_ex(x,pred_x,y)

            m_sx_np_loss = np_loss_sx(x,pred_x,y)
            m_sx_intp_loss = intp_loss_sx(x,pred_x,y)
            m_sx_bulk_loss = bulk_loss_sx(x,pred_x,y)

            total_eyz = m_eyz_np_loss + m_eyz_bulk_loss + m_eyz_intp_loss
            total_ex = m_ex_np_loss + m_ex_bulk_loss + m_ex_intp_loss
            total_sx = m_sx_np_loss + m_sx_bulk_loss + m_sx_intp_loss
             

            total_loss = e_x_loss + e_yz_loss + s_x_loss + total_eyz + total_ex + total_sx
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.e_x_loss_tracker.update_state(e_x_loss)
        self.e_yz_loss_tracker.update_state(e_yz_loss)
        self.s_x_loss_tracker.update_state(s_x_loss)
        self.e_x_global_loss_tracker.update_state(total_ex)
        self.e_yz_global_loss_tracker.update_state(total_eyz)
        self.s_x_global_loss_tracker.update_state(total_sx)        
        return {
            "loss": self.total_loss_tracker.result(),
            "e_x_loss": self.e_x_loss_tracker.result(),
            "e_yz_loss": self.e_yz_loss_tracker.result(),
            "s_x_loss": self.s_x_loss_tracker.result(),
            "e_x_global_loss": self.e_x_global_loss_tracker.result(),
            "e_yz_global_loss": self.e_yz_global_loss_tracker.result(),
            "s_x_global_loss": self.s_x_global_loss_tracker.result(),   
        }

    def test_step(self, input_data):
      x, y = input_data
      pred_x = self.NN(x) 
      e_x_loss = tf.keras.losses.MeanSquaredError()(pred_x[:,:,0], y[:,:,0])
      e_yz_loss = tf.keras.losses.MeanSquaredError()(pred_x[:,:,1], y[:,:,1])
      s_x_loss = tf.keras.losses.MeanSquaredError()(pred_x[:,:,2], y[:,:,2])
      m_eyz_np_loss = np_loss_eyz(x,pred_x,y)
      m_eyz_intp_loss = intp_loss_eyz(x,pred_x,y)
      m_eyz_bulk_loss = bulk_loss_eyz(x,pred_x,y)

      m_ex_np_loss = np_loss_ex(x,pred_x,y)
      m_ex_intp_loss = intp_loss_ex(x,pred_x,y)
      m_ex_bulk_loss = bulk_loss_ex(x,pred_x,y)

      m_sx_np_loss = np_loss_sx(x,pred_x,y)
      m_sx_intp_loss = intp_loss_sx(x,pred_x,y)
      m_sx_bulk_loss = bulk_loss_sx(x,pred_x,y)

      total_eyz = m_eyz_np_loss + m_eyz_bulk_loss + m_eyz_intp_loss
      total_ex = m_ex_np_loss + m_ex_bulk_loss + m_ex_intp_loss
      total_sx = m_sx_np_loss + m_sx_bulk_loss + m_sx_intp_loss
             

      val_total_loss = e_x_loss + e_yz_loss + s_x_loss + total_eyz + total_ex + total_sx      

      return {"loss": val_total_loss,
              "e_x_loss": e_x_loss,
              "e_yz_loss": e_yz_loss,
              "s_x_loss": s_x_loss,
              "e_x_global_loss": total_ex,
              "e_yz_global_loss": total_eyz,
              "s_x_global_loss": total_sx,

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
    save_freq='epoch', save_best_only=False, period=1)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.8, min_lr=0.000001)
    
model = DNN(Generator)
model.compile(optimizer=tf.keras.optimizers.Adam())


def plot_stress_strain_multi_all(pred_strain_mean,pred_strain_std,target_strain_mean,target_strain_std,pred_stress_mean,pred_stress_std,target_stress_mean,target_stress_std, labels, save_path, legend,path):

    colors = ['r', 'g', 'b']
    linestyles = ['-.', '--', ':']
    plt.figure(figsize=(8, 6))
    alpha_pred = 0.4
    alpha_tar = 0.1
    vols = [4.5, 12.7, 16.1]

    for i in range(len(pred_stress_mean)):
        # Flatten all stress data and compute the minimum value
        all_stress_mean = np.concatenate((target_stress_mean[i], pred_stress_mean[i]))
        all_stress_std = np.concatenate((target_stress_std[i], pred_stress_std[i]))
            
        min_y = np.min(all_stress_mean-all_stress_std)
        y_offset = -min_y if min_y < 0 else 0
        # Apply offset if necessary
        target_shifted = target_stress_mean[i] + y_offset
        pred_shifted = pred_stress_mean[i] + y_offset

        # Plot target
        plt.plot(target_strain_mean[i], target_shifted, linestyle=linestyles[i], color=colors[i],
                 label=rf"${{\phi}}(\%)={vols[i]}$ Target", linewidth=2)
        plt.fill_between(target_strain_mean[i], target_shifted-target_stress_std[i], target_shifted+target_stress_std[i], color=colors[i], alpha=alpha_tar)
        # Plot prediction
        plt.plot(pred_strain_mean[i], pred_shifted, linestyle='-', color=colors[i],
                 label=rf"${{\phi}}(\%)={vols[i]}$ Prediction", linewidth=2)
        plt.fill_between(pred_strain_mean[i], pred_shifted-target_stress_std[i], pred_shifted+pred_stress_std[i], color=colors[i], alpha=alpha_pred)
        
    plt.xlabel(labels[0], fontsize=28)
    plt.ylabel(labels[1], fontsize=28)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 0.06)  # Force x-axis range to show up to 0.06
    if legend:
     plt.legend(fontsize=18, frameon=True)
    else: 
     plt.legend(fontsize=18, frameon=True).remove()
        
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(path+save_path+".jpg", dpi=300, bbox_inches='tight')

    plt.close()


def plot_curves(target_mean,target_std,pred_mean,pred_std,labels,path):
    
     x1 = labels[0]
     y1_1 = labels[1]
     y1_2 = labels[2]
     
     x2 = labels[3]
     y2_1 = labels[4]
     y2_2 = labels[5]
     
     x3 = labels[6]
     y3_1 = labels[7]
     y3_2 = labels[8]
     
     x4 = labels[9]
     y4_1 = labels[10]
     y4_2 = labels[11]

     volume = labels[12]     
     
     alpha = 0.3
     width = 3.   
     alpha_target = 0.1
     alpha_pred = 0.4
     axis_font = 40
     tick_font = 25
     col_stress_tar = '#e41a1c'
     col_stress_pred = '#e41a1c'
     col_strain_tar = '#377eb8'
     col_strain_pred = '#377eb8'
     
     fig = plt.figure(layout='constrained', figsize=(22, 15))

     ax= fig.subplots(2,2)
     
     y_pred_strain_mean = pred_mean[8]  
     x_pred_strain_mean = pred_mean[5]
     y_tar_strain_mean = target_mean[8]    
     x_tar_strain_mean = target_mean[5]

     y_pred_stress_mean = pred_mean[11]
     x_pred_stress_mean = pred_mean[5]
     y_tar_stress_mean = target_mean[11]    
     x_tar_stress_mean = target_mean[5]

     y_pred_strain_std = pred_std[8]  
     x_pred_strain_std = pred_std[5]
     y_tar_strain_std = target_std[8]    
     x_tar_strain_std = target_std[5]

     y_pred_stress_std = pred_std[11]
     x_pred_stress_std = pred_std[5]
     y_tar_stress_std = target_std[11]    
     x_tar_stress_std = target_std[5]
     
     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean-y_tar_stress_std])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s

     ax[0,0].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[0,0].yaxis.label.set_color(col_strain_pred)
     ax[0,0].set_facecolor("yellow")
     ax[0,0].patch.set_alpha(alpha) 

     
     ax[0,0].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[0,0].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[0,0].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle=":", linewidth=width)   
     ax[0,0].fill_between(x_tar_strain_mean, y_tar_strain_mean-y_tar_strain_std, y_tar_strain_mean+y_tar_strain_std, color=col_strain_tar, alpha=alpha_target)
     
     ax[0,0].set_xlabel(x1, fontsize=axis_font)
     ax[0,0].set_ylabel(y1_1, fontsize=axis_font)
     
     ax01 = ax[0,0].twinx()
     ax01.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax01.yaxis.label.set_color(col_stress_pred)
     ax01.set_ylabel(y1_2, fontsize=axis_font)

     ax01.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax01.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax01.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle=":", linewidth=width)   
     ax01.fill_between(x_tar_stress_mean, y_tar_stress_mean-y_tar_stress_std, y_tar_stress_mean+y_tar_stress_std, color=col_stress_tar, alpha=alpha_target)


     

     y_pred_strain_mean = pred_mean[7]  
     x_pred_strain_mean = pred_mean[4]
     y_tar_strain_mean = target_mean[7]    
     x_tar_strain_mean = target_mean[4]

     y_pred_stress_mean = pred_mean[10]
     x_pred_stress_mean = pred_mean[4]
     y_tar_stress_mean = target_mean[10]    
     x_tar_stress_mean = target_mean[4]

     y_pred_strain_std = pred_std[7]  
     x_pred_strain_std = pred_std[4]
     y_tar_strain_std = target_std[7]    
     x_tar_strain_std = target_std[4]

     y_pred_stress_std = pred_std[10]
     x_pred_stress_std = pred_std[4]
     y_tar_stress_std = target_std[10]    
     x_tar_stress_std = target_std[4]

     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean-y_tar_stress_std])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s

     
     ax[1,0].set_facecolor("blue")
     ax[1,0].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[1,0].yaxis.label.set_color(col_strain_pred)
     ax[1,0].patch.set_alpha(0.1)
     ax[1,0].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[1,0].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[1,0].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
     ax[1,0].fill_between(x_tar_strain_mean, y_tar_strain_mean-y_tar_strain_std, y_tar_strain_mean+y_tar_strain_std, color=col_strain_tar, alpha=alpha_target)

    
     ax[1,0].set_xlabel(x2, fontsize=axis_font)
     ax[1,0].set_ylabel(y2_1, fontsize=axis_font)
   
     
     ax02 = ax[1,0].twinx()
     ax02.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax02.yaxis.label.set_color(col_stress_pred)
     ax02.set_ylabel(y2_2, fontsize=axis_font)
     ax02.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax02.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax02.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
     ax02.fill_between(x_tar_stress_mean, y_tar_stress_mean-y_tar_stress_std, y_tar_stress_mean+y_tar_stress_std, color=col_stress_tar, alpha=alpha_target)
     
     
     
     y_pred_strain_mean = pred_mean[6]  
     x_pred_strain_mean = pred_mean[3]
     y_tar_strain_mean = target_mean[6]    
     x_tar_strain_mean = target_mean[3]

     y_pred_stress_mean = pred_mean[9]
     x_pred_stress_mean = pred_mean[3]
     y_tar_stress_mean = target_mean[9]    
     x_tar_stress_mean = target_mean[3]

     y_pred_strain_std = pred_std[6]  
     x_pred_strain_std = pred_std[3]
     y_tar_strain_std = target_std[6]    
     x_tar_strain_std = target_std[3]

     y_pred_stress_std = pred_std[9]
     x_pred_stress_std = pred_std[3]
     y_tar_stress_std = target_std[9]    
     x_tar_stress_std = target_std[3]

     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean-y_tar_stress_std])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s


     ax[0,1].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[0,1].yaxis.label.set_color(col_strain_pred)
     ax[0,1].set_facecolor("orange")
     ax[0,1].patch.set_alpha(alpha)
     ax[0,1].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[0,1].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[0,1].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
     ax[0,1].fill_between(x_tar_strain_mean, y_tar_strain_mean-y_tar_strain_std, y_tar_strain_mean+y_tar_strain_std, color=col_strain_tar, alpha=alpha_target)
    
     ax[0,1].set_xlabel(x3, fontsize=axis_font)
     ax[0,1].set_ylabel(y3_1, fontsize=axis_font)
      
     
     ax03 = ax[0,1].twinx()
     ax03.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax03.yaxis.label.set_color(col_stress_pred)
     ax03.set_ylabel(y3_2, fontsize=axis_font)
     ax03.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax03.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax03.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
     ax03.fill_between(x_tar_stress_mean, y_tar_stress_mean-y_tar_stress_std, y_tar_stress_mean+y_tar_stress_std, color=col_stress_tar, alpha=alpha_target)

     y_pred_strain_mean = pred_mean[1]  
     x_pred_strain_mean = pred_mean[0]
     y_tar_strain_mean = target_mean[1]    
     x_tar_strain_mean = target_mean[0]

     y_pred_stress_mean = pred_mean[2]
     x_pred_stress_mean = pred_mean[0]
     y_tar_stress_mean = target_mean[2]    
     x_tar_stress_mean = target_mean[0]

     y_pred_strain_std = pred_std[1]  
     x_pred_strain_std = pred_std[0]
     y_tar_strain_std = target_std[1]    
     x_tar_strain_std = target_std[0]

     y_pred_stress_std = pred_std[2]
     x_pred_stress_std = pred_std[0]
     y_tar_stress_std = target_std[2]    
     x_tar_stress_std = target_std[0]

     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean-y_tar_stress_std])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s


     ax[1,1].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[1,1].yaxis.label.set_color(col_strain_pred)
     ax[1,1].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[1,1].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[1,1].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
     ax[1,1].fill_between(x_tar_strain_mean, y_tar_strain_mean-y_tar_strain_std, y_tar_strain_mean+y_tar_strain_std, color=col_strain_tar, alpha=alpha_target)
    
     ax[1,1].set_xlabel(x4, fontsize=axis_font)
     ax[1,1].set_ylabel(y4_1, fontsize=axis_font)
      
     
     ax04 = ax[1,1].twinx()
     ax04.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax04.yaxis.label.set_color(col_stress_pred)
     ax04.set_ylabel(y4_2, fontsize=axis_font)
     ax04.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax04.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax04.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
     ax04.fill_between(x_tar_stress_mean, y_tar_stress_mean-y_tar_stress_std, y_tar_stress_mean+y_tar_stress_std, color=col_stress_tar, alpha=alpha_target)

     
     fig.savefig(path+'figure_'+str(volume)+'.jpg', bbox_inches='tight',dpi=300)
     plt.close(fig)  

    
     return


def plot_curves_without_target_int(target_mean,pred_mean,pred_std,labels,path):
    
     x1 = labels[0]
     y1_1 = labels[1]
     y1_2 = labels[2]
     
     x2 = labels[3]
     y2_1 = labels[4]
     y2_2 = labels[5]
     
     x3 = labels[6]
     y3_1 = labels[7]
     y3_2 = labels[8]
     
     x4 = labels[9]
     y4_1 = labels[10]
     y4_2 = labels[11]

     volume = labels[12]     
     
     alpha = 0.3
     alpha_pred = 0.4
     width = 3.     
     axis_font = 40
     tick_font = 25
     col_stress_tar = '#e41a1c'
     col_stress_pred = '#e41a1c'
     col_strain_tar = '#377eb8'
     col_strain_pred = '#377eb8'
     
     fig = plt.figure(layout='constrained', figsize=(22, 15))

     ax= fig.subplots(2,2)

     y_pred_strain_mean = pred_mean[8]  
     x_pred_strain_mean = pred_mean[5]
     y_tar_strain_mean = target_mean[8]    
     x_tar_strain_mean = target_mean[5]

     y_pred_stress_mean = pred_mean[11]
     x_pred_stress_mean = pred_mean[5]
     y_tar_stress_mean = target_mean[11]    
     x_tar_stress_mean = target_mean[5]

     y_pred_strain_std = pred_std[8]  
     x_pred_strain_std = pred_std[5]

     y_pred_stress_std = pred_std[11]
     x_pred_stress_std = pred_std[5]
     
     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s

     ax[0,0].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[0,0].yaxis.label.set_color(col_strain_pred)
     ax[0,0].set_facecolor("yellow")
     ax[0,0].patch.set_alpha(alpha) 

     
     ax[0,0].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[0,0].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[0,0].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle=":", linewidth=width)   
     
     ax[0,0].set_xlabel(x1, fontsize=axis_font)
     ax[0,0].set_ylabel(y1_1, fontsize=axis_font)
     
     ax01 = ax[0,0].twinx()
     ax01.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax01.yaxis.label.set_color(col_stress_pred)
     ax01.set_ylabel(y1_2, fontsize=axis_font)

     ax01.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax01.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax01.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle=":", linewidth=width)   


     

     y_pred_strain_mean = pred_mean[7]  
     x_pred_strain_mean = pred_mean[4]
     y_tar_strain_mean = target_mean[7]    
     x_tar_strain_mean = target_mean[4]

     y_pred_stress_mean = pred_mean[10]
     x_pred_stress_mean = pred_mean[4]
     y_tar_stress_mean = target_mean[10]    
     x_tar_stress_mean = target_mean[4]

     y_pred_strain_std = pred_std[7]  
     x_pred_strain_std = pred_std[4]


     y_pred_stress_std = pred_std[10]
     x_pred_stress_std = pred_std[4]


     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s

     
     ax[1,0].set_facecolor("blue")
     ax[1,0].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[1,0].yaxis.label.set_color(col_strain_pred)
     ax[1,0].patch.set_alpha(0.1)
     ax[1,0].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[1,0].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[1,0].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   


    
     ax[1,0].set_xlabel(x2, fontsize=axis_font)
     ax[1,0].set_ylabel(y2_1, fontsize=axis_font)
   
     
     ax02 = ax[1,0].twinx()
     ax02.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax02.yaxis.label.set_color(col_stress_pred)
     ax02.set_ylabel(y2_2, fontsize=axis_font)
     ax02.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax02.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax02.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   

     
     
     
     y_pred_strain_mean = pred_mean[6]  
     x_pred_strain_mean = pred_mean[3]
     y_tar_strain_mean = target_mean[6]    
     x_tar_strain_mean = target_mean[3]

     y_pred_stress_mean = pred_mean[9]
     x_pred_stress_mean = pred_mean[3]
     y_tar_stress_mean = target_mean[9]    
     x_tar_stress_mean = target_mean[3]

     y_pred_strain_std = pred_std[6]  
     x_pred_strain_std = pred_std[3]


     y_pred_stress_std = pred_std[9]
     x_pred_stress_std = pred_std[3]


     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s


     ax[0,1].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[0,1].yaxis.label.set_color(col_strain_pred)
     ax[0,1].set_facecolor("orange")
     ax[0,1].patch.set_alpha(alpha)
     ax[0,1].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[0,1].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[0,1].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   

    
     ax[0,1].set_xlabel(x3, fontsize=axis_font)
     ax[0,1].set_ylabel(y3_1, fontsize=axis_font)
      
     
     ax03 = ax[0,1].twinx()
     ax03.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax03.yaxis.label.set_color(col_stress_pred)
     ax03.set_ylabel(y3_2, fontsize=axis_font)
     ax03.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax03.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax03.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   

    
     y_pred_strain_mean = pred_mean[1]  
     x_pred_strain_mean = pred_mean[0]
     y_tar_strain_mean = target_mean[1]    
     x_tar_strain_mean = target_mean[0]

     y_pred_stress_mean = pred_mean[2]
     x_pred_stress_mean = pred_mean[0]
     y_tar_stress_mean = target_mean[2]    
     x_tar_stress_mean = target_mean[0]

     y_pred_strain_std = pred_std[1]  
     x_pred_strain_std = pred_std[0]


     y_pred_stress_std = pred_std[2]
     x_pred_stress_std = pred_std[0]


     min_s = np.min([y_pred_stress_mean-y_pred_stress_std,y_tar_stress_mean])
     
     y_pred_stress_mean -= min_s
     y_tar_stress_mean -= min_s


     ax[1,1].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[1,1].yaxis.label.set_color(col_strain_pred)
     ax[1,1].plot(x_pred_strain_mean, y_pred_strain_mean, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[1,1].fill_between(x_pred_strain_mean, y_pred_strain_mean-y_pred_strain_std, y_pred_strain_mean+y_pred_strain_std, color=col_strain_pred, alpha=alpha_pred)

     ax[1,1].plot(x_tar_strain_mean, y_tar_strain_mean, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   

     ax[1,1].set_xlabel(x4, fontsize=axis_font)
     ax[1,1].set_ylabel(y4_1, fontsize=axis_font)
      
     
     ax04 = ax[1,1].twinx()
     ax04.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax04.yaxis.label.set_color(col_stress_pred)
     ax04.set_ylabel(y4_2, fontsize=axis_font)
     ax04.plot(x_pred_stress_mean, y_pred_stress_mean, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax04.fill_between(x_pred_stress_mean, y_pred_stress_mean-y_pred_stress_std, y_pred_stress_mean+y_pred_stress_std, color=col_stress_pred, alpha=alpha_pred)

     ax04.plot(x_tar_stress_mean, y_tar_stress_mean, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   

    
     
     fig.savefig(path+'figure_'+str(volume)+'.jpg', bbox_inches='tight',dpi=300)
     plt.close(fig)  

    
     return


def plot_scatters_stress(data_list,labels,fig,y1,y2,r2):     
     alpha = 0.3
     axis_font = 55
     tick_font = 25
     width = 5.
     marker_size=60.
     lw = 0.3
     alpha_int = 0.7
     markers = ["o", "s", "^", "D", "x"]     
     ax1,ax2,ax3,ax4 = fig.subplots(4,1)

     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
     data_list = np.array(data_list)
     min_s = np.min([data_list[:,1],data_list[:,0]])
     
     data_list[:,1] -= min_s
     data_list[:,0] -= min_s     

     ax1.tick_params(axis='y', labelsize= tick_font)
     ax1.set_facecolor("yellow")
     ax1.patch.set_alpha(alpha) 
     ax1.scatter(data_list[0,1],data_list[0,0], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax1.scatter(data_list[1,1],data_list[1,0], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax1.scatter(data_list[2,1],data_list[2,0], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax1.scatter(data_list[3,1],data_list[3,0], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax1.scatter(data_list[4,1],data_list[4,0], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
     np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
     ]

     ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=width)
     ax1.set_aspect('equal')
     ax1.set_xlim(lims)
     ax1.set_ylim(lims)
     ax1.set_xlabel("Predicted "+labels[0], fontsize=axis_font)
     if y1:
      ax1.set_ylabel("Target "+labels[0], fontsize=axis_font)
      ax1.text(0.05, 0.95, r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[3][0],r2[3][1]), transform=ax1.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)


     min_s = np.min([data_list[:,3],data_list[:,2]])
     
     data_list[:,3] -= min_s
     data_list[:,2] -= min_s             
     
     ax2.set_facecolor("blue")
     ax2.tick_params(axis='y', labelsize= tick_font)
     ax2.patch.set_alpha(0.1)
     ax2.scatter(data_list[0,3],data_list[0,2], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax2.scatter(data_list[1,3],data_list[1,2], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax2.scatter(data_list[2,3],data_list[2,2], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax2.scatter(data_list[3,3],data_list[3,2], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax2.scatter(data_list[4,3],data_list[4,2], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
     np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
     ]

     ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=width)
     ax2.set_aspect('equal')
     ax2.set_xlim(lims)
     ax2.set_ylim(lims)
     ax2.set_xlabel("Predicted "+labels[1], fontsize=axis_font)
     if y1: 
      ax2.set_ylabel("Target "+labels[1], fontsize=axis_font)
      ax2.text(0.05, 0.95, r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[2][0],r2[2][1]), transform=ax2.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)
     
     
     min_s = np.min([data_list[:,4],data_list[:,5]])
     
     data_list[:,4] -= min_s
     data_list[:,5] -= min_s     
     
     
     ax3.tick_params(axis='y', labelsize= tick_font)
     ax3.set_facecolor("orange")
     ax3.patch.set_alpha(alpha)
     ax3.scatter(data_list[0,5],data_list[0,4], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax3.scatter(data_list[1,5],data_list[1,4], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax3.scatter(data_list[2,5],data_list[2,4], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax3.scatter(data_list[3,5],data_list[3,4], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax3.scatter(data_list[4,5],data_list[4,4], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax3.get_xlim(), ax3.get_ylim()]),  # min of both axes
     np.max([ax3.get_xlim(), ax3.get_ylim()]),  # max of both axes
     ]

     ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
     ax3.set_aspect('equal')
     ax3.set_xlim(lims)
     ax3.set_ylim(lims)
     ax3.set_xlabel("Predicted "+labels[2], fontsize=axis_font)
     if y1:
      ax3.set_ylabel("Target "+labels[2], fontsize=axis_font)
      ax3.text(0.05, 0.95, r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[1][0],r2[1][1]), transform=ax3.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)
     
     min_s = np.min([data_list[:,7],data_list[:,6]])
     
     data_list[:,7] -= min_s
     data_list[:,6] -= min_s     
     
     
     ax4.tick_params(axis='y', labelsize= tick_font)
     ax4.scatter(data_list[0,7],data_list[0,6], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax4.scatter(data_list[1,7],data_list[1,6], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax4.scatter(data_list[2,7],data_list[2,6], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax4.scatter(data_list[3,7],data_list[3,6], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax4.scatter(data_list[4,7],data_list[4,6], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
     np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
     ]

     ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=width)
     ax4.set_aspect('equal')
     ax4.set_xlim(lims)
     ax4.set_ylim(lims)
     ax4.set_xlabel("Predicted "+labels[3], fontsize=axis_font)
     if y1:
      ax4.set_ylabel("Target "+labels[3], fontsize=axis_font)
      ax4.text(0.05, 0.95,r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[0][0],r2[0][1]), transform=ax4.transAxes, fontsize=axis_font,
         verticalalignment='top', bbox=props)
     
     return

def plot_scatters(data_list,labels,fig,y1,y2,r2):     
     alpha = 0.3
     axis_font = 55
     tick_font = 25
     width = 5.
     marker_size=60.
     lw = 0.3
     alpha_int = 0.7
     markers = ["o", "s", "^", "D", "x"]
     ax1,ax2,ax3,ax4 = fig.subplots(4,1)

     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
      
     ax1.tick_params(axis='y', labelsize= tick_font)
     ax1.set_facecolor("yellow")
     ax1.patch.set_alpha(alpha) 
     ax1.scatter(data_list[0][1],data_list[0][0], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax1.scatter(data_list[1][1],data_list[1][0], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax1.scatter(data_list[2][1],data_list[2][0], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax1.scatter(data_list[3][1],data_list[3][0], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax1.scatter(data_list[4][1],data_list[4][0], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
     np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
     ]

     ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=width)
     ax1.set_aspect('equal')
     ax1.set_xlim(lims)
     ax1.set_ylim(lims)
     ax1.set_xlabel("Predicted "+labels[0], fontsize=axis_font)
     if y1:
      ax1.set_ylabel("Target "+labels[0], fontsize=axis_font)
      ax1.text(0.05, 0.95, r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[3][0],r2[3][1]), transform=ax1.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)

     
     
     ax2.set_facecolor("blue")
     ax2.tick_params(axis='y', labelsize= tick_font)
     ax2.patch.set_alpha(0.1)
     ax2.scatter(data_list[0][3],data_list[0][2], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax2.scatter(data_list[1][3],data_list[1][2], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax2.scatter(data_list[2][3],data_list[2][2], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax2.scatter(data_list[3][3],data_list[3][2], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax2.scatter(data_list[4][3],data_list[4][2], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
     np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
     ]

     ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=width)
     ax2.set_aspect('equal')
     ax2.set_xlim(lims)
     ax2.set_ylim(lims)
     ax2.set_xlabel("Predicted "+labels[1], fontsize=axis_font)
     if y1: 
      ax2.set_ylabel("Target "+labels[1], fontsize=axis_font)
      ax2.text(0.05, 0.95, r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[2][0],r2[2][1]), transform=ax2.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)
     
     
     
     
     ax3.tick_params(axis='y', labelsize= tick_font)
     ax3.set_facecolor("orange")
     ax3.patch.set_alpha(alpha)
     ax3.scatter(data_list[0][5],data_list[0][4], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax3.scatter(data_list[1][5],data_list[1][4], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax3.scatter(data_list[2][5],data_list[2][4], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax3.scatter(data_list[3][5],data_list[3][4], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax3.scatter(data_list[4][5],data_list[4][4], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax3.get_xlim(), ax3.get_ylim()]),  # min of both axes
     np.max([ax3.get_xlim(), ax3.get_ylim()]),  # max of both axes
     ]

     ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
     ax3.set_aspect('equal')
     ax3.set_xlim(lims)
     ax3.set_ylim(lims)
     ax3.set_xlabel("Predicted "+labels[2], fontsize=axis_font)
     if y1:
      ax3.set_ylabel("Target "+labels[2], fontsize=axis_font)
      ax3.text(0.05, 0.95, r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[1][0],r2[1][1]), transform=ax3.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)
     
     
     
     ax4.tick_params(axis='y', labelsize= tick_font)
     ax4.scatter(data_list[0][7],data_list[0][6], s=marker_size, linewidth=lw, marker=markers[0], alpha=alpha_int)  
     ax4.scatter(data_list[1][7],data_list[1][6], s=marker_size, linewidth=lw, marker=markers[1], alpha=alpha_int)  
     ax4.scatter(data_list[2][7],data_list[2][6], s=marker_size, linewidth=lw, marker=markers[2], alpha=alpha_int)  
     ax4.scatter(data_list[3][7],data_list[3][6], s=marker_size, linewidth=lw, marker=markers[3], alpha=alpha_int)  
     ax4.scatter(data_list[4][7],data_list[4][6], s=marker_size, linewidth=lw, marker=markers[4], alpha=alpha_int)  
     lims = [
     np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
     np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
     ]

     ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=width)
     ax4.set_aspect('equal')
     ax4.set_xlim(lims)
     ax4.set_ylim(lims)
     ax4.set_xlabel("Predicted "+labels[3], fontsize=axis_font)
     if y1:
      ax4.set_ylabel("Target "+labels[3], fontsize=axis_font)
      ax4.text(0.05, 0.95,r"$R^2 = {0:.2g} \pm {1:.2f}$".format(r2[0][0],r2[0][1]), transform=ax4.transAxes, fontsize=axis_font,
         verticalalignment='top', bbox=props)
     
     return

def compute_mean_std(data_list):
    return np.mean(data_list),np.std(data_list)

def plot_multi_scatters(data,labels,r2,path):
     
     fig = plt.figure(layout='constrained', figsize=(33, 40))
     subfigs = fig.subfigures(1, 3, width_ratios = (1.,1.,1.))
    
     data_ex_1 = [data[0][0],data[0][2],data[0][8],data[0][10],data[0][16],data[0][18],data[0][24],data[0][26]]
     data_elat_1 = [data[0][1],data[0][3],data[0][9],data[0][11],data[0][17],data[0][19],data[0][25],data[0][27]]
     data_sx_1 = [data[0][5],data[0][7],data[0][13],data[0][15],data[0][21],data[0][23],data[0][29],data[0][31]]

     data_ex_2 = [data[1][0],data[1][2],data[1][8],data[1][10],data[1][16],data[1][18],data[1][24],data[1][26]]
     data_elat_2 = [data[1][1],data[1][3],data[1][9],data[1][11],data[1][17],data[1][19],data[1][25],data[1][27]]
     data_sx_2 = [data[1][5],data[1][7],data[1][13],data[1][15],data[1][21],data[1][23],data[1][29],data[1][31]]

     data_ex_3 = [data[2][0],data[2][2],data[2][8],data[2][10],data[2][16],data[2][18],data[2][24],data[2][26]]
     data_elat_3 = [data[2][1],data[2][3],data[2][9],data[2][11],data[2][17],data[2][19],data[2][25],data[2][27]]
     data_sx_3 = [data[2][5],data[2][7],data[2][13],data[2][15],data[2][21],data[2][23],data[2][29],data[2][31]]

     data_ex_4 = [data[3][0],data[3][2],data[3][8],data[3][10],data[3][16],data[3][18],data[3][24],data[3][26]]
     data_elat_4 = [data[3][1],data[3][3],data[3][9],data[3][11],data[3][17],data[3][19],data[3][25],data[3][27]]
     data_sx_4 = [data[3][5],data[3][7],data[3][13],data[3][15],data[3][21],data[3][23],data[3][29],data[3][31]]

     data_ex_5 = [data[4][0],data[4][2],data[4][8],data[4][10],data[4][16],data[4][18],data[4][24],data[4][26]]
     data_elat_5 = [data[4][1],data[4][3],data[4][9],data[4][11],data[4][17],data[4][19],data[4][25],data[4][27]]
     data_sx_5 = [data[4][5],data[4][7],data[4][13],data[4][15],data[4][21],data[4][23],data[4][29],data[4][31]]

     data_ex = [data_ex_1,data_ex_2,data_ex_3,data_ex_4,data_ex_5]     
     data_elat = [data_elat_1,data_elat_2,data_elat_3,data_elat_4,data_elat_5]     
     data_sx = [data_sx_1,data_sx_2,data_sx_3,data_sx_4,data_sx_5]     


     labels_ex = [labels[0],labels[3],labels[6],labels[9]]
     labels_elat = [labels[1],labels[4],labels[7],labels[10]]
     labels_sx = [labels[2],labels[5],labels[8],labels[11]]

     r2_ex_g = compute_mean_std([r2[0][0],r2[1][0],r2[2][0],r2[3][0],r2[4][0]])
     r2_elat_g = compute_mean_std([r2[0][1],r2[1][1],r2[2][1],r2[3][1],r2[4][1]])
     r2_sx_g = compute_mean_std([r2[0][2],r2[1][2],r2[2][2],r2[3][2],r2[4][2]])
     r2_ex_bulk = compute_mean_std([r2[0][3],r2[1][3],r2[2][3],r2[3][3],r2[4][3]])
     r2_ex_int = compute_mean_std([r2[0][4],r2[1][4],r2[2][4],r2[3][4],r2[4][4]])
     r2_ex_np = compute_mean_std([r2[0][5],r2[1][5],r2[2][5],r2[3][5],r2[4][5]])
     r2_elat_bulk = compute_mean_std([r2[0][6],r2[1][6],r2[2][6],r2[3][6],r2[4][6]])
     r2_elat_int = compute_mean_std([r2[0][7],r2[1][7],r2[2][7],r2[3][7],r2[4][7]])
     r2_elat_np = compute_mean_std([r2[0][8],r2[1][8],r2[2][8],r2[3][8],r2[4][8]])
     r2_sx_bulk = compute_mean_std([r2[0][9],r2[1][9],r2[2][9],r2[3][9],r2[4][9]])
     r2_sx_int = compute_mean_std([r2[0][10],r2[1][10],r2[2][10],r2[3][10],r2[4][10]])
     r2_sx_np = compute_mean_std([r2[0][11],r2[1][11],r2[2][11],r2[3][11],r2[4][11]])

     r2_ex = [r2_ex_g,r2_ex_bulk,r2_ex_int,r2_ex_np]
     r2_elat = [r2_elat_g,r2_elat_bulk,r2_elat_int,r2_elat_np]
     r2_sx = [r2_sx_g,r2_sx_bulk,r2_sx_int,r2_sx_np]
     
     plot_scatters(data_list=data_ex, labels=labels_ex, fig=subfigs[0],y1=True,y2=True,r2=r2_ex)
     plot_scatters(data_list=data_elat, labels=labels_elat, fig=subfigs[1],y1=True,y2=True,r2=r2_elat)     
     plot_scatters_stress(data_list=data_sx, labels=labels_sx, fig=subfigs[2],y1=True, y2=True,r2=r2_sx)
     
     # fig.tight_layout()
     fig.savefig(path+'figure_scatters'+str(labels[-1])+'.jpg', bbox_inches='tight',dpi=300)
     
     plt.close(fig)  
     return





def model_evaluation(model,test_input,test_target,epoch,volume,r,run_path):
    
    
    checkpoint_path = "./tmps/model_"+ str(run_path)+ ".h5"
   
    n_batches = int(test_input.shape[0])
    model.built = True
    model.load_weights(checkpoint_path)

    test_input = tf.cast(test_input,dtype=tf.float32)
    test_target = tf.cast(test_target,dtype=tf.float32)
    
    pred_global_e_x_list = []
    tar_global_e_x_list = []


    pred_global_e_x_bulk = []
    tar_global_e_x_bulk = []

    pred_global_e_x_intp = []
    tar_global_e_x_intp = []
    
    pred_global_e_x_np = []
    tar_global_e_x_np = []   


    pred_global_e_yz_bulk = []
    tar_global_e_yz_bulk = []

    pred_global_e_yz_intp = []
    tar_global_e_yz_intp = []
    
    pred_global_e_yz_np = []
    tar_global_e_yz_np = []    


    pred_global_e_yz_list = []
    tar_global_e_yz_list = []


    pred_global_s_x_list = []
    tar_global_s_x_list = []

    pred_global_s_x_bulk = []
    tar_global_s_x_bulk = []

    pred_global_s_x_intp = []
    tar_global_s_x_intp = []
    
    pred_global_s_x_np = []
    tar_global_s_x_np = []    


    
    V_int = (4./3.)*np.pi*(25.5**3 - 19.5**3)
    V_np = (4./3.)*np.pi*(19.5**3)
    V_bulk = r**3 - (4./3.)*np.pi*(25.5**3)
    V_box = r**3
    total_time = 0
    for i in range(n_batches):
        start_time = time.time()
        pred = model.NN.predict(test_input[i:i+1],verbose=0)
        total_time += (time.time() - start_time)
        # print("--- %s seconds ---" % (time.time() - start_time))
        pred_e_x = pred[0,:,:,0]
        pred_e_yz = pred[0,:,:,1]
        pred_s_x = pred[0,:,:,2]

        pred_s_x *= 10000000.    


        ind_ex = tf.where(tf.not_equal(test_target[i,:,:,0],0))
        ind_elat = tf.where(tf.not_equal(test_target[i,:,:,1],0))
        ind_sx = tf.where(tf.not_equal(test_target[i,:,:,2],0))

        pred_global_e_x = tf.reduce_mean(tf.gather_nd(pred_e_x[:,:],ind_ex))
        global_e_x = tf.reduce_mean(tf.gather_nd(test_target[i,:,:,0],ind_ex))

        pred_global_e_yz = tf.reduce_mean(tf.gather_nd(pred_e_yz[:,:],ind_elat))
        global_e_yz = tf.reduce_mean(tf.gather_nd(test_target[i,:,:,1],ind_elat))

        
        pred_global_e_x_list.append(pred_global_e_x.numpy())
        tar_global_e_x_list.append(global_e_x.numpy())


        pred_global_e_yz_list.append(pred_global_e_yz.numpy())
        tar_global_e_yz_list.append(global_e_yz.numpy())
        
        ind_np = tf.where(tf.equal(test_input[i,:,:,-7],1))
        ind_intf = tf.where(tf.equal(test_input[i,:,:,-6],1))
        ind_bulk = tf.where(tf.equal(test_input[i,:,:,-5],1))
        
        
        s_x_bulk_pred = tf.reduce_sum(tf.gather_nd(pred_s_x,ind_bulk)).numpy()/(V_bulk*10000)
        s_x_int_pred = tf.reduce_sum(tf.gather_nd(pred_s_x,ind_intf)).numpy()/(V_int*10000)
        s_x_np_pred = tf.reduce_sum(tf.gather_nd(pred_s_x,ind_np)).numpy()/(V_np*10000)

        s_x_bulk_tar = tf.reduce_sum(tf.gather_nd(test_target[i,:,:,2],ind_bulk)).numpy()/(V_bulk*10000)
        s_x_int_tar = tf.reduce_sum(tf.gather_nd(test_target[i,:,:,2],ind_intf)).numpy()/(V_int*10000)
        s_x_np_tar = tf.reduce_sum(tf.gather_nd(test_target[i,:,:,2],ind_np)).numpy()/(V_np*10000)

        
        pred_global_s_x_np.append(s_x_np_pred)
        pred_global_s_x_intp.append(s_x_int_pred)
        pred_global_s_x_bulk.append(s_x_bulk_pred)
        
        tar_global_s_x_np.append(s_x_np_tar)
        tar_global_s_x_intp.append(s_x_int_tar)
        tar_global_s_x_bulk.append(s_x_bulk_tar)

        
        pred_global_s_x = (s_x_int_pred*V_int+s_x_bulk_pred*V_bulk+s_x_np_pred*V_np)/V_box
        global_s_x = (s_x_int_tar*V_int+s_x_bulk_tar*V_bulk+s_x_np_tar*V_np)/V_box
        

        pred_global_s_x_list.append(pred_global_s_x)
        tar_global_s_x_list.append(global_s_x)

        
        pred_global_e_yz_np.append(tf.reduce_mean(tf.gather_nd(pred_e_yz,ind_np)).numpy())
        pred_global_e_yz_intp.append(tf.reduce_mean(tf.gather_nd(pred_e_yz,ind_intf)).numpy())
        pred_global_e_yz_bulk.append(tf.reduce_mean(tf.gather_nd(pred_e_yz,ind_bulk)).numpy())
        
        tar_global_e_yz_np.append(tf.reduce_mean(tf.gather_nd(test_target[i,:,:,1],ind_np)).numpy())
        tar_global_e_yz_intp.append(tf.reduce_mean(tf.gather_nd(test_target[i,:,:,1],ind_intf)).numpy())
        tar_global_e_yz_bulk.append(tf.reduce_mean(tf.gather_nd(test_target[i,:,:,1],ind_bulk)).numpy())
        
        
        pred_global_e_x_np.append(tf.reduce_mean(tf.gather_nd(pred_e_x,ind_np)).numpy())
        pred_global_e_x_intp.append(tf.reduce_mean(tf.gather_nd(pred_e_x,ind_intf)).numpy())
        pred_global_e_x_bulk.append(tf.reduce_mean(tf.gather_nd(pred_e_x,ind_bulk)).numpy())
        
        tar_global_e_x_np.append(tf.reduce_mean(tf.gather_nd(test_target[i,:,:,0],ind_np)).numpy())
        tar_global_e_x_intp.append(tf.reduce_mean(tf.gather_nd(test_target[i,:,:,0],ind_intf)).numpy())
        tar_global_e_x_bulk.append(tf.reduce_mean(tf.gather_nd(test_target[i,:,:,0],ind_bulk)).numpy())       
        


    np_list =[tar_global_e_x_np,tar_global_e_yz_np,pred_global_e_x_np,pred_global_e_yz_np,tar_global_e_x_np,tar_global_s_x_np,pred_global_e_x_np,pred_global_s_x_np
               ,tar_global_e_x_intp,tar_global_e_yz_intp,pred_global_e_x_intp,pred_global_e_yz_intp,tar_global_e_x_intp,tar_global_s_x_intp,pred_global_e_x_intp,pred_global_s_x_intp,
               tar_global_e_x_bulk,tar_global_e_yz_bulk,pred_global_e_x_bulk,pred_global_e_yz_bulk,tar_global_e_x_bulk,tar_global_s_x_bulk,pred_global_e_x_bulk,pred_global_s_x_bulk,
               tar_global_e_x_list,tar_global_e_yz_list,pred_global_e_x_list,pred_global_e_yz_list,tar_global_e_x_list,tar_global_s_x_list,pred_global_e_x_list,pred_global_s_x_list]
    r2_list = [r2_score(tar_global_e_x_list,pred_global_e_x_list),r2_score(tar_global_e_yz_list,pred_global_e_yz_list),r2_score(tar_global_s_x_list,pred_global_s_x_list),
               r2_score(tar_global_e_x_bulk,pred_global_e_x_bulk),r2_score(tar_global_e_x_intp,pred_global_e_x_intp),r2_score(tar_global_e_x_np,pred_global_e_x_np),
               r2_score(tar_global_e_yz_bulk,pred_global_e_yz_bulk),r2_score(tar_global_e_yz_intp,pred_global_e_yz_intp),r2_score(tar_global_e_yz_np,pred_global_e_yz_np),
               r2_score(tar_global_s_x_bulk,pred_global_s_x_bulk),r2_score(tar_global_s_x_intp,pred_global_s_x_intp),r2_score(tar_global_s_x_np,pred_global_s_x_np)]
    np_labels = [r"$\mathbf{{\epsilon}_{x}^{NP}}$",r"$\mathbf{{\epsilon}_{lat}^{NP}}$",r"$\mathbf{{\sigma}_{x}^{NP}}$ (GPa)",
                  r"$\mathbf{{\epsilon}_{x}^{I}}$",r"$\mathbf{{\epsilon}_{lat}^{I}}$",r"$\mathbf{{\sigma}_{x}^{I}}$ (GPa)",
                  r"$\mathbf{{\epsilon}_{x}^{M}}$",r"$\mathbf{{\epsilon}_{lat}^{M}}$",r"$\mathbf{{\sigma}_{x}^{M}}$ (GPa)",
                  r"$\mathbf{\epsilon_{x}^{global}}$",r"$\mathbf{\epsilon_{lat}^{global}}$",r"$\mathbf{\sigma_{x}^{global}}$ (GPa)",volume]
       
    
    #print(total_time/n_batches)    
    return  np_list, np_labels, r2_list   





def compute_target_all(test_input,test_target,volume,r):

    n_frames = 100
    n_confs = int(test_input.shape[0]/n_frames)
    avg_frames = n_confs
    
    test_input = tf.cast(test_input,dtype=tf.float32)
    test_target = tf.cast(test_target,dtype=tf.float32)

    V_int = (4./3.)*np.pi*(25.5**3 - 19.5**3)
    V_np = (4./3.)*np.pi*(19.5**3)
    V_bulk = r**3 - (4./3.)*np.pi*(25.5**3)
    V_box = r**3

    global_e_x_bulk = []
    global_e_x_intp = []
    global_e_x_np = []   
    global_e_yz_bulk = []
    global_e_yz_intp = []
    global_e_yz_np = []  
    global_s_x_bulk = []
    global_s_x_intp = []
    global_s_x_np = []   
    global_e_x_ar = []
    global_e_lat_ar = []
    global_s_x_ar = [] 

    
    
    for i_conf in range(n_confs):
      global_e_x_bulk_list = []
      global_e_x_intp_list =  []
      global_e_x_np_list =  []   
      global_e_yz_bulk_list = []
      global_e_yz_intp_list = []
      global_e_yz_np_list = []    
      global_s_x_bulk_list = []
      global_s_x_intp_list = []
      global_s_x_np_list = []    
      global_e_x_list = []
      global_e_lat_list = []
      global_s_x_list = []  

      
      for i_frame in range(n_frames):
         
        idx = i_frame + i_conf*n_frames  
        
        
        ind_np = tf.where(tf.equal(test_input[idx,:,:,-7],1))
        ind_intf = tf.where(tf.equal(test_input[idx,:,:,-6],1))
        ind_bulk = tf.where(tf.equal(test_input[idx,:,:,-5],1))
        
        
        ind_ex = tf.where(tf.not_equal(test_target[idx,:,:,0],0))
        ind_elat = tf.where(tf.not_equal(test_target[idx,:,:,1],0))
        
        global_e_x = tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,0],ind_ex))
        global_e_lat = tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,1],ind_elat))
        
        s_x_np = tf.reduce_sum(tf.gather_nd(test_target[idx,:,:,2],ind_np)).numpy()/(V_np*10000)
        s_x_bulk = tf.reduce_sum(tf.gather_nd(test_target[idx,:,:,2],ind_bulk)).numpy()/(V_bulk*10000)
        s_x_int = tf.reduce_sum(tf.gather_nd(test_target[idx,:,:,2],ind_intf)).numpy()/(V_int*10000)
        
        global_s_x_np_list.append(s_x_np)
        global_s_x_intp_list.append(s_x_int)
        global_s_x_bulk_list.append(s_x_bulk)


        global_e_yz_np_list.append(tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,1],ind_np)).numpy())
        global_e_yz_intp_list.append(tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,1],ind_intf)).numpy())
        global_e_yz_bulk_list.append(tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,1],ind_bulk)).numpy())
                
        global_e_x_np_list.append(tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,0],ind_np)).numpy())
        global_e_x_intp_list.append(tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,0],ind_intf)).numpy())
        global_e_x_bulk_list.append(tf.reduce_mean(tf.gather_nd(test_target[idx,:,:,0],ind_bulk)).numpy())       
        
        global_s_x = (s_x_int*V_int+s_x_bulk*V_bulk+s_x_np*V_np)/V_box
        
        
        global_s_x_list.append(global_s_x)
        global_e_x_list.append(global_e_x)
        global_e_lat_list.append(global_e_lat)


      global_s_x_bulk.append(global_s_x_bulk_list)
      global_s_x_np.append(global_s_x_np_list)
      global_s_x_intp.append(global_s_x_intp_list)

      global_e_x_bulk.append(global_e_x_bulk_list)
      global_e_x_np.append(global_e_x_np_list)
      global_e_x_intp.append(global_e_x_intp_list)

      global_e_yz_bulk.append(global_e_yz_bulk_list)
      global_e_yz_np.append(global_e_yz_np_list)
      global_e_yz_intp.append(global_e_yz_intp_list)

      global_s_x_ar.append(global_s_x_list)
      global_e_x_ar.append(global_e_x_list)
      global_e_lat_ar.append(global_e_lat_list)

   


    mean_e_x_bulk,std_e_x_bulk = np.mean(np.array(global_e_x_bulk),axis=0), np.std(np.array(global_e_x_bulk),axis=0)
    mean_e_x_intp,std_e_x_intp = np.mean(np.array(global_e_x_intp),axis=0), np.std(np.array(global_e_x_intp),axis=0)
    mean_e_x_np,std_e_x_np = np.mean(np.array(global_e_x_np),axis=0), np.std(np.array(global_e_x_np),axis=0)

    mean_s_x_bulk,std_s_x_bulk = np.mean(np.array(global_s_x_bulk),axis=0), np.std(np.array(global_s_x_bulk),axis=0)
    mean_s_x_intp,std_s_x_intp = np.mean(np.array(global_s_x_intp),axis=0), np.std(np.array(global_s_x_intp),axis=0)
    mean_s_x_np,std_s_x_np = np.mean(np.array(global_s_x_np),axis=0), np.std(np.array(global_s_x_np),axis=0)

    mean_e_yz_bulk,std_e_yz_bulk = np.mean(np.array(global_e_yz_bulk),axis=0), np.std(np.array(global_e_yz_bulk),axis=0)
    mean_e_yz_intp,std_e_yz_intp = np.mean(np.array(global_e_yz_intp),axis=0), np.std(np.array(global_e_yz_intp),axis=0)
    mean_e_yz_np,std_e_yz_np = np.mean(np.array(global_e_yz_np),axis=0), np.std(np.array(global_e_yz_np),axis=0)
   
    mean_e_x,std_e_x = np.mean(np.array(global_e_x_ar),axis=0), np.std(np.array(global_e_x_ar),axis=0)
    mean_e_yz,std_e_yz = np.mean(np.array(global_e_lat_ar),axis=0), np.std(np.array(global_e_lat_ar),axis=0)
    mean_s_x,std_s_x = np.mean(np.array(global_s_x_ar),axis=0), np.std(np.array(global_s_x_ar),axis=0)

    std_list = [std_e_x,std_e_yz,std_s_x,std_e_x_bulk,std_e_x_intp,std_e_x_np,std_e_yz_bulk,std_e_yz_intp,std_e_yz_np,std_s_x_bulk,std_s_x_intp,std_s_x_np]    
    mean_list = [mean_e_x,mean_e_yz,mean_s_x,mean_e_x_bulk,mean_e_x_intp,mean_e_x_np,mean_e_yz_bulk,mean_e_yz_intp,mean_e_yz_np,mean_s_x_bulk,mean_s_x_intp,mean_s_x_np]    

    return mean_list, std_list

def compute_mean_std_all(data_list):
    idx_list = [26,27,31,18,10,2,19,11,3,23,15,7] #[ex,elat,sx,exm,exi,exnp,elatm,elati,elatnp,sxm,sxi,sxnp]
    global_e_x_np =[]
    global_e_x_bulk = []
    global_e_x_intp = []
    global_e_x = []
    global_s_x_np =[]
    global_s_x_bulk = []
    global_s_x_intp = []
    global_s_x = []
    global_e_yz_np =[]
    global_e_yz_bulk = []
    global_e_yz_intp = []
    global_e_lat = []
    for i in range(5):
        global_e_x.append(data_list[i][idx_list[0]])
        global_e_lat.append(data_list[i][idx_list[1]])
        global_s_x.append(data_list[i][idx_list[2]])
        global_e_x_bulk.append(data_list[i][idx_list[3]])
        global_e_x_intp.append(data_list[i][idx_list[4]])
        global_e_x_np.append(data_list[i][idx_list[5]])
        global_e_yz_bulk.append(data_list[i][idx_list[6]])
        global_e_yz_intp.append(data_list[i][idx_list[7]])
        global_e_yz_np.append(data_list[i][idx_list[8]])
        global_s_x_bulk.append(data_list[i][idx_list[9]])
        global_s_x_intp.append(data_list[i][idx_list[10]])
        global_s_x_np.append(data_list[i][idx_list[11]])
    
    mean_e_x_bulk,std_e_x_bulk = np.mean(np.array(global_e_x_bulk),axis=0), np.std(np.array(global_e_x_bulk),axis=0)
    mean_e_x_intp,std_e_x_intp = np.mean(np.array(global_e_x_intp),axis=0), np.std(np.array(global_e_x_intp),axis=0)
    mean_e_x_np,std_e_x_np = np.mean(np.array(global_e_x_np),axis=0), np.std(np.array(global_e_x_np),axis=0)

    mean_s_x_bulk,std_s_x_bulk = np.mean(np.array(global_s_x_bulk),axis=0), np.std(np.array(global_s_x_bulk),axis=0)
    mean_s_x_intp,std_s_x_intp = np.mean(np.array(global_s_x_intp),axis=0), np.std(np.array(global_s_x_intp),axis=0)
    mean_s_x_np,std_s_x_np = np.mean(np.array(global_s_x_np),axis=0), np.std(np.array(global_s_x_np),axis=0)

    mean_e_yz_bulk,std_e_yz_bulk = np.mean(np.array(global_e_yz_bulk),axis=0), np.std(np.array(global_e_yz_bulk),axis=0)
    mean_e_yz_intp,std_e_yz_intp = np.mean(np.array(global_e_yz_intp),axis=0), np.std(np.array(global_e_yz_intp),axis=0)
    mean_e_yz_np,std_e_yz_np = np.mean(np.array(global_e_yz_np),axis=0), np.std(np.array(global_e_yz_np),axis=0)
   
    mean_e_x,std_e_x = np.mean(np.array(global_e_x),axis=0), np.std(np.array(global_e_x),axis=0)
    mean_e_yz,std_e_yz = np.mean(np.array(global_e_lat),axis=0), np.std(np.array(global_e_lat),axis=0)
    mean_s_x,std_s_x = np.mean(np.array(global_s_x),axis=0), np.std(np.array(global_s_x),axis=0)
    
    std_list = [std_e_x,std_e_yz,std_s_x,std_e_x_bulk,std_e_x_intp,std_e_x_np,std_e_yz_bulk,std_e_yz_intp,std_e_yz_np,std_s_x_bulk,std_s_x_intp,std_s_x_np]    
    mean_list = [mean_e_x,mean_e_yz,mean_s_x,mean_e_x_bulk,mean_e_x_intp,mean_e_x_np,mean_e_yz_bulk,mean_e_yz_intp,mean_e_yz_np,mean_s_x_bulk,mean_s_x_intp,mean_s_x_np]    

    return mean_list, std_list

v1 = 12.7
v2 = 16.1
v4 = 4.5
v5 = 7.6

r1 = 58
r2 = 52
r4 = 84
r5 = 69


epoch_list = [1500]
for i in epoch_list:
 with open(PATH+'/test_target_1.pkl','rb') as f:
    test_target_2 = pickle.load(f)
    print(test_target_2.shape)


 with open(PATH+'/test_input_1.pkl','rb') as f:
    test_input_2 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(test_input_2.shape) 



 with open(PATH+'/test_target_2.pkl','rb') as f:
    test_target_1 = pickle.load(f)
    print(test_target_1.shape)


 with open(PATH+'/test_input_2.pkl','rb') as f:
    test_input_1 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(test_input_1.shape) 
    
   

 with open(PATH+'/test_target_4.pkl','rb') as f:
    test_target_4 = pickle.load(f)
    print(test_target_4.shape)


 with open(PATH+'/test_input_4.pkl','rb') as f:
    test_input_4 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(test_input_4.shape) 


 with open(PATH+'/test_target_5.pkl','rb') as f:
    test_target_5 = pickle.load(f)
    print(test_target_5.shape)


 with open(PATH+'/test_input_5.pkl','rb') as f:
    test_input_5 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(test_input_5.shape) 

 epoch = i

 PATH_plots = "./plots_"+str(i)+"/"
 run_path = ''
 
 data_1_1,labels_1,r_1_1 = model_evaluation(model,test_input_1,test_target_1,epoch,v2,r2,run_path+str(1))   
 data_2_1,labels_2,r_2_1 = model_evaluation(model,test_input_2,test_target_2,epoch,v1,r1,run_path+str(1))
 data_4_1,labels_4,r_4_1 = model_evaluation(model,test_input_4,test_target_4,epoch,v4,r4,run_path+str(1))
 data_5_1,labels_5,r_5_1 = model_evaluation(model,test_input_5,test_target_5,epoch,v5,r5,run_path+str(1))

 data_1_2,labels_1,r_1_2 = model_evaluation(model,test_input_1,test_target_1,epoch,v2,r2,run_path+str(2))   
 data_2_2,labels_2,r_2_2 = model_evaluation(model,test_input_2,test_target_2,epoch,v1,r1,run_path+str(2))
 data_4_2,labels_4,r_4_2 = model_evaluation(model,test_input_4,test_target_4,epoch,v4,r4,run_path+str(2))
 data_5_2,labels_5,r_5_2 = model_evaluation(model,test_input_5,test_target_5,epoch,v5,r5,run_path+str(2))

 data_1_3,labels_1,r_1_3 = model_evaluation(model,test_input_1,test_target_1,epoch,v2,r2,run_path+str(3))   
 data_2_3,labels_2,r_2_3 = model_evaluation(model,test_input_2,test_target_2,epoch,v1,r1,run_path+str(3))
 data_4_3,labels_4,r_4_3 = model_evaluation(model,test_input_4,test_target_4,epoch,v4,r4,run_path+str(3))
 data_5_3,labels_5,r_5_3 = model_evaluation(model,test_input_5,test_target_5,epoch,v5,r5,run_path+str(3))

 data_1_4,labels_1,r_1_4 = model_evaluation(model,test_input_1,test_target_1,epoch,v2,r2,run_path+str(4))   
 data_2_4,labels_2,r_2_4 = model_evaluation(model,test_input_2,test_target_2,epoch,v1,r1,run_path+str(4))
 data_4_4,labels_4,r_4_4 = model_evaluation(model,test_input_4,test_target_4,epoch,v4,r4,run_path+str(4))
 data_5_4,labels_5,r_5_4 = model_evaluation(model,test_input_5,test_target_5,epoch,v5,r5,run_path+str(4))

 data_1_5,labels_1,r_1_5 = model_evaluation(model,test_input_1,test_target_1,epoch,v2,r2,run_path+str(5))   
 data_2_5,labels_2,r_2_5 = model_evaluation(model,test_input_2,test_target_2,epoch,v1,r1,run_path+str(5))
 data_4_5,labels_4,r_4_5 = model_evaluation(model,test_input_4,test_target_4,epoch,v4,r4,run_path+str(5))
 data_5_5,labels_5,r_5_5 = model_evaluation(model,test_input_5,test_target_5,epoch,v5,r5,run_path+str(5))

 plot_multi_scatters([data_1_1,data_1_2,data_1_3,data_1_4,data_1_5], labels_1, [r_1_1,r_1_2,r_1_3,r_1_4,r_1_5],PATH_plots)
 plot_multi_scatters([data_2_1,data_2_2,data_2_3,data_2_4,data_2_5], labels_2, [r_2_1,r_2_2,r_2_3,r_2_4,r_2_5],PATH_plots)
 plot_multi_scatters([data_4_1,data_4_2,data_4_3,data_4_4,data_4_5], labels_4, [r_4_1,r_4_2,r_4_3,r_4_4,r_4_5],PATH_plots)
 plot_multi_scatters([data_5_1,data_5_2,data_5_3,data_5_4,data_5_5], labels_5, [r_5_1,r_5_2,r_5_3,r_5_4,r_5_5],PATH_plots)

 with open(PATH+'/val_target_1.pkl','rb') as f:
    val_target_2 = pickle.load(f)
    print(val_target_2.shape)


 with open(PATH+'/val_input_1.pkl','rb') as f:
    val_input_2 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(val_input_2.shape) 



 with open(PATH+'/val_target_2.pkl','rb') as f:
    val_target_1 = pickle.load(f)
    print(val_target_1.shape)


 with open(PATH+'/val_input_2.pkl','rb') as f:
    val_input_1 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(val_input_1.shape) 
    
   

 with open(PATH+'/val_target_4.pkl','rb') as f:
    val_target_4 = pickle.load(f)
    print(val_target_4.shape)


 with open(PATH+'/val_input_4.pkl','rb') as f:
    val_input_4 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(val_input_4.shape) 


 with open(PATH+'/train_target_1.pkl','rb') as f:
    train_target_2 = pickle.load(f)
    print(train_target_2.shape)


 with open(PATH+'/train_input_1.pkl','rb') as f:
    train_input_2 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(train_input_2.shape) 



 with open(PATH+'/train_target_2.pkl','rb') as f:
    train_target_1 = pickle.load(f)
    print(train_target_1.shape)


 with open(PATH+'/train_input_2.pkl','rb') as f:
    train_input_1 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(train_input_1.shape) 
    
   

 with open(PATH+'/train_target_4.pkl','rb') as f:
    train_target_4 = pickle.load(f)
    print(train_target_4.shape)


 with open(PATH+'/train_input_4.pkl','rb') as f:
    train_input_4 = pickle.load(f)[:,:,:,[0,1,2,6,7,8,9,10,11,12]]
    print(train_input_4.shape) 

 
 input_1 = np.concatenate((train_input_1,val_input_1,test_input_1),axis=0)
 target_1 = np.concatenate((train_target_1,val_target_1,test_target_1),axis=0)
 
 input_2 = np.concatenate((train_input_2,val_input_2,test_input_2),axis=0)
 target_2 = np.concatenate((train_target_2,val_target_2,test_target_2),axis=0)

 input_4 = np.concatenate((train_input_4,val_input_4,test_input_4),axis=0)
 target_4 = np.concatenate((train_target_4,val_target_4,test_target_4),axis=0)

 target_mean_1,target_std_1 = compute_target_all(input_1,target_1,v2,r2)
 target_mean_2,target_std_2= compute_target_all(input_2,target_2,v1,r1)
 target_mean_4,target_std_4 = compute_target_all(input_4,target_4,v4,r4)
 target_mean_5,target_std_5 = compute_target_all(test_input_5,test_target_5,v5,r5)
 
 pred_mean_1,pred_std_1 = compute_mean_std_all([data_1_1,data_1_2,data_1_3,data_1_4,data_1_5])
 pred_mean_2,pred_std_2 = compute_mean_std_all([data_2_1,data_2_2,data_2_3,data_2_4,data_2_5])
 pred_mean_4,pred_std_4 = compute_mean_std_all([data_4_1,data_4_2,data_4_3,data_4_4,data_4_5])
 pred_mean_5,pred_std_5 = compute_mean_std_all([data_5_1,data_5_2,data_5_3,data_5_4,data_5_5])
 
 
 plot_stress_strain_multi_all([pred_mean_4[4],pred_mean_2[4],pred_mean_1[4]],[pred_std_4[4],pred_std_2[4],pred_std_1[4]], [target_mean_4[4],target_mean_2[4],target_mean_1[4]], [target_std_4[4],target_std_2[4],target_std_1[4]],   [pred_mean_4[10],pred_mean_2[10],pred_mean_1[10]],[pred_std_4[10],pred_std_2[10],pred_std_1[10]], [target_mean_4[10],target_mean_2[10],target_mean_1[10]],[target_std_4[10],target_std_2[10],target_std_1[10]],[r"$\mathbf{{\epsilon}_{x}^{I}}$",r"$\mathbf{{\sigma}_{x}^{I}}$"],"int_stress",False,PATH_plots)
 plot_stress_strain_multi_all([pred_mean_4[3],pred_mean_2[3],pred_mean_1[3]],[pred_std_4[3],pred_std_2[3],pred_std_1[3]], [target_mean_4[3],target_mean_2[3],target_mean_1[3]], [target_std_4[3],target_std_2[3],target_std_1[3]],   [pred_mean_4[9],pred_mean_2[9],pred_mean_1[9]],[pred_std_4[9],pred_std_2[9],pred_std_1[9]], [target_mean_4[9],target_mean_2[9],target_mean_1[9]],[target_std_4[9],target_std_2[9],target_std_1[9]],[r"$\mathbf{{\epsilon}_{x}^{M}}$",r"$\mathbf{{\sigma}_{x}^{M}}$"],"bulk_stress",True,PATH_plots)
 plot_stress_strain_multi_all([pred_mean_4[0],pred_mean_2[0],pred_mean_1[0]],[pred_std_4[0],pred_std_2[0],pred_std_1[0]], [target_mean_4[0],target_mean_2[0],target_mean_1[0]], [target_std_4[0],target_std_2[0],target_std_1[0]],   [pred_mean_4[2],pred_mean_2[2],pred_mean_1[2]],[pred_std_4[2],pred_std_2[2],pred_std_1[2]], [target_mean_4[2],target_mean_2[2],target_mean_1[2]],[target_std_4[2],target_std_2[2],target_std_1[2]],[r"$\mathbf{{\epsilon}_{x}^{global}}$",r"$\mathbf{{\sigma}_{x}^{global}}$"],"global_stress",False,PATH_plots)
 
 
 plot_curves(target_mean_1,target_std_1,pred_mean_1,pred_std_1,labels_1,PATH_plots)
 plot_curves(target_mean_2,target_std_2,pred_mean_2,pred_std_2,labels_2,PATH_plots)
 plot_curves(target_mean_4,target_std_4,pred_mean_4,pred_std_4,labels_4,PATH_plots)
 plot_curves_without_target_int(target_mean_5,pred_mean_5,pred_std_5,labels_5,PATH_plots) 
 
 