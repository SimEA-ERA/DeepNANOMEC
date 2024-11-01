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


def plot_scatter(tar,pred,labels,path):
    # title = labels[0]
    x = labels[0]
    y = labels[1]
    save = labels[2]
    fig, ax = plt.subplots()
    plt.figure()
    ax.scatter(pred,tar)

    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # ax.set_title(title)
    ax.set_xlabel(x, fontsize=25)
    ax.set_ylabel(y, fontsize=25)
    fig.savefig(path+'scatter_'+save+'.png', bbox_inches='tight',dpi=300)
    plt.close(fig)


def plot_curves_5(data_list,labels,path):
    
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
     width = 5.     
     axis_font = 40
     tick_font = 25
     col_stress_tar = '#e41a1c'
     col_stress_pred = '#e41a1c'#"#b33dc6"
     col_strain_tar = '#377eb8'
     col_strain_pred = '#377eb8'
     
     fig = plt.figure(layout='constrained', figsize=(22, 15))
     # fig.title(r'$\phi = %1.1f $' %volume,fontsize=50)

     ax= fig.subplots(2,2)

     y_pred_strain = [x for _,x in sorted(zip(data_list[2],data_list[3]))] 
     x_pred_strain = sorted(data_list[2])
     y_tar_strain = [x for _,x in sorted(zip(data_list[0],data_list[1]))]     
     x_tar_strain = sorted(data_list[0])

     y_pred_stress = [x for _,x in sorted(zip(data_list[6],data_list[7]))] 
     x_pred_stress = sorted(data_list[6])
     y_tar_stress = [x for _,x in sorted(zip(data_list[4],data_list[5]))]     
     x_tar_stress = sorted(data_list[4])


     ax[0,0].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[0,0].yaxis.label.set_color(col_strain_pred)
     ax[0,0].set_facecolor("yellow")
     ax[0,0].patch.set_alpha(alpha) 
     ax[0,0].plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[0,0].plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax[0,0].set_xlabel(x1, fontsize=axis_font)
     ax[0,0].set_ylabel(y1_1, fontsize=axis_font)
     
     ax01 = ax[0,0].twinx()
     ax01.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax01.yaxis.label.set_color(col_stress_pred)
     ax01.set_ylabel(y1_2, fontsize=axis_font)
     ax01.plot(x_pred_stress, y_pred_stress, label="Prediction", color= col_stress_pred,linestyle="solid", linewidth=width)   
     ax01.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar ,linestyle="dotted", linewidth=width)   
     
     
     
     y_pred_strain = [x for _,x in sorted(zip(data_list[10],data_list[11]))] 
     x_pred_strain = sorted(data_list[10])
     y_tar_strain = [x for _,x in sorted(zip(data_list[8],data_list[9]))]     
     x_tar_strain = sorted(data_list[8])

     y_pred_stress = [x for _,x in sorted(zip(data_list[14],data_list[15]))] 
     x_pred_stress = sorted(data_list[14])
     y_tar_stress = [x for _,x in sorted(zip(data_list[12],data_list[13]))]     
     x_tar_stress = sorted(data_list[12])

     ax[1,0].set_facecolor("blue")
     ax[1,0].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[1,0].yaxis.label.set_color(col_strain_pred)
     ax[1,0].patch.set_alpha(0.1)
     ax[1,0].plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[1,0].plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax[1,0].set_xlabel(x2, fontsize=axis_font)
     ax[1,0].set_ylabel(y2_1, fontsize=axis_font)
   
     
     ax02 = ax[1,0].twinx()
     ax02.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax02.yaxis.label.set_color(col_stress_pred)
     ax02.set_ylabel(y2_2, fontsize=axis_font)
     ax02.plot(x_pred_stress, y_pred_stress, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax02.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
     
     
     
     y_pred_strain = [x for _,x in sorted(zip(data_list[18],data_list[19]))] 
     x_pred_strain = sorted(data_list[18])
     y_tar_strain = [x for _,x in sorted(zip(data_list[16],data_list[17]))]     
     x_tar_strain = sorted(data_list[16])

     y_pred_stress = [x for _,x in sorted(zip(data_list[22],data_list[23]))] 
     x_pred_stress = sorted(data_list[22])
     y_tar_stress = [x for _,x in sorted(zip(data_list[20],data_list[21]))]     
     x_tar_stress = sorted(data_list[20])

     ax[0,1].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[0,1].yaxis.label.set_color(col_strain_pred)
     ax[0,1].set_facecolor("orange")
     ax[0,1].patch.set_alpha(alpha)
     ax[0,1].plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[0,1].plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax[0,1].set_xlabel(x3, fontsize=axis_font)
     ax[0,1].set_ylabel(y3_1, fontsize=axis_font)
      
     
     ax03 = ax[0,1].twinx()
     ax03.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax03.yaxis.label.set_color(col_stress_pred)
     ax03.set_ylabel(y3_2, fontsize=axis_font)
     ax03.plot(x_pred_stress, y_pred_stress, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax03.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
    
     y_pred_strain = [x for _,x in sorted(zip(data_list[26],data_list[27]))] 
     x_pred_strain = sorted(data_list[26])
     y_tar_strain = [x for _,x in sorted(zip(data_list[24],data_list[25]))]     
     x_tar_strain = sorted(data_list[24])

     y_pred_stress = [x for _,x in sorted(zip(data_list[30],data_list[31]))] 
     x_pred_stress = sorted(data_list[30])
     y_tar_stress = [x for _,x in sorted(zip(data_list[28],data_list[29]))]     
     x_tar_stress = sorted(data_list[28])

     ax[1,1].tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax[1,1].yaxis.label.set_color(col_strain_pred)
     ax[1,1].plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax[1,1].plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax[1,1].set_xlabel(x4, fontsize=axis_font)
     ax[1,1].set_ylabel(y4_1, fontsize=axis_font)
      
     
     ax04 = ax[1,1].twinx()
     ax04.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax04.yaxis.label.set_color(col_stress_pred)
     ax04.set_ylabel(y4_2, fontsize=axis_font)
     ax04.plot(x_pred_stress, y_pred_stress, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax04.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
    
     
     fig.savefig(path+'figure_'+str(volume)+'.jpg', bbox_inches='tight',dpi=300)
     plt.close(fig)  

    
     return


def plot_scatters(data_list,labels,fig,y1,y2):     
     alpha = 0.3
     axis_font = 55
     tick_font = 25
     width = 5.
     marker_size=50.
     ax1,ax2,ax3,ax4 = fig.subplots(4,1)

     props = dict(boxstyle='round', facecolor='white', alpha=0.5)


     # ax1.set_title(r'$\phi = %1.1f $' %volume,fontsize=60, pad=15)


     ax1.tick_params(axis='y', labelsize= tick_font)
     ax1.set_facecolor("yellow")
     ax1.patch.set_alpha(alpha) 
     ax1.scatter(data_list[1],data_list[0], s=marker_size)   
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
      ax1.text(0.05, 0.95, r"R$^2$ = {0:.2g}".format(r2_score(data_list[0],data_list[1])), transform=ax1.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)

     
     
     ax2.set_facecolor("blue")
     ax2.tick_params(axis='y', labelsize= tick_font)
     ax2.patch.set_alpha(0.1)
     ax2.scatter(data_list[3],data_list[2], s=marker_size)   
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
      ax2.text(0.05, 0.95, r"R$^2$ = {0:.2g}".format(r2_score(data_list[2],data_list[3])), transform=ax2.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)
     
     
     
     
     ax3.tick_params(axis='y', labelsize= tick_font)
     ax3.set_facecolor("orange")
     ax3.patch.set_alpha(alpha)
     ax3.scatter(data_list[5],data_list[4], s=marker_size)
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
      ax3.text(0.05, 0.95, r"R$^2$ = {0:.2g}".format(r2_score(data_list[4],data_list[5])), transform=ax3.transAxes, fontsize=axis_font,
        verticalalignment='top', bbox=props)
     
     
     
     ax4.tick_params(axis='y', labelsize= tick_font)
     ax4.scatter(data_list[7],data_list[6], s=marker_size)
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
      ax4.text(0.05, 0.95,r"R$^2$ = {0:.2g}".format(r2_score(data_list[6],data_list[7])), transform=ax4.transAxes, fontsize=axis_font,
         verticalalignment='top', bbox=props)
     
     return


def plot_multi_scatters(data,labels,path):
     
     fig = plt.figure(layout='constrained', figsize=(33, 40))
     subfigs = fig.subfigures(1, 3, width_ratios = (1.,1.,1.))
    
     data_ex = [data[0],data[2],data[8],data[10],data[16],data[18],data[24],data[26]]
     data_elat = [data[1],data[3],data[9],data[11],data[17],data[19],data[25],data[27]]
     data_sx = [data[5],data[7],data[13],data[15],data[21],data[23],data[29],data[31]]
     
     labels_ex = [labels[0],labels[3],labels[6],labels[9]]
     labels_elat = [labels[1],labels[4],labels[7],labels[10]]
     labels_sx = [labels[2],labels[5],labels[8],labels[11]]
     
     
     plot_scatters(data_list=data_ex, labels=labels_ex, fig=subfigs[0],y1=True,y2=True)
     plot_scatters(data_list=data_elat, labels=labels_elat, fig=subfigs[1],y1=True,y2=True)     
     plot_scatters(data_list=data_sx, labels=labels_sx, fig=subfigs[2],y1=True, y2=True)
     
     # fig.tight_layout()
     fig.savefig(path+'figure_scatters'+str(labels[-1])+'.jpg', bbox_inches='tight',dpi=300)
     
     plt.close(fig)  
     return



def plot_curves(data_list,labels,fig,y1,y2):
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
     width = 5.     
     axis_font = 55
     tick_font = 25
     col_stress_tar = '#e41a1c'
     col_stress_pred = '#e41a1c'#"#b33dc6"
     col_strain_tar = '#377eb8'
     col_strain_pred = '#377eb8'
     

     ax1,ax2,ax3,ax4 = fig.subplots(4,1)


     ax1.set_title(r'$\phi = %1.1f $' %volume,fontsize=60, pad=15)

     y_pred_strain = [x for _,x in sorted(zip(data_list[2],data_list[3]))] 
     x_pred_strain = sorted(data_list[2])
     y_tar_strain = [x for _,x in sorted(zip(data_list[0],data_list[1]))]     
     x_tar_strain = sorted(data_list[0])

     y_pred_stress = [x for _,x in sorted(zip(data_list[6],data_list[7]))] 
     x_pred_stress = sorted(data_list[6])
     y_tar_stress = [x for _,x in sorted(zip(data_list[4],data_list[5]))]     
     x_tar_stress = sorted(data_list[4])


     ax1.tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax1.yaxis.label.set_color(col_strain_pred)
     ax1.set_facecolor("yellow")
     ax1.patch.set_alpha(alpha) 
     ax1.plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax1.plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax1.set_xlabel(x1, fontsize=axis_font)
     if y1:
      ax1.set_ylabel(y1_1, fontsize=axis_font)
     
     ax01 = ax1.twinx()
     ax01.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax01.yaxis.label.set_color(col_stress_pred)
     if y2:
      ax01.set_ylabel(y1_2, fontsize=axis_font)
     ax01.plot(x_pred_stress, y_pred_stress, label="Prediction", color= col_stress_pred,linestyle="solid", linewidth=width)   
     ax01.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar ,linestyle="dotted", linewidth=width)   
     
     
     
     y_pred_strain = [x for _,x in sorted(zip(data_list[10],data_list[11]))] 
     x_pred_strain = sorted(data_list[10])
     y_tar_strain = [x for _,x in sorted(zip(data_list[8],data_list[9]))]     
     x_tar_strain = sorted(data_list[8])

     y_pred_stress = [x for _,x in sorted(zip(data_list[14],data_list[15]))] 
     x_pred_stress = sorted(data_list[14])
     y_tar_stress = [x for _,x in sorted(zip(data_list[12],data_list[13]))]     
     x_tar_stress = sorted(data_list[12])

     ax2.set_facecolor("blue")
     ax2.tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax2.yaxis.label.set_color(col_strain_pred)
     ax2.patch.set_alpha(0.1)
     ax2.plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax2.plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax2.set_xlabel(x2, fontsize=axis_font)
     if y1: 
      ax2.set_ylabel(y2_1, fontsize=axis_font)
   
     
     ax02 = ax2.twinx()
     ax02.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax02.yaxis.label.set_color(col_stress_pred)
     if y2: 
      ax02.set_ylabel(y2_2, fontsize=axis_font)
     ax02.plot(x_pred_stress, y_pred_stress, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax02.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
     
     
     
     y_pred_strain = [x for _,x in sorted(zip(data_list[18],data_list[19]))] 
     x_pred_strain = sorted(data_list[18])
     y_tar_strain = [x for _,x in sorted(zip(data_list[16],data_list[17]))]     
     x_tar_strain = sorted(data_list[16])

     y_pred_stress = [x for _,x in sorted(zip(data_list[22],data_list[23]))] 
     x_pred_stress = sorted(data_list[22])
     y_tar_stress = [x for _,x in sorted(zip(data_list[20],data_list[21]))]     
     x_tar_stress = sorted(data_list[20])

     ax3.tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax3.yaxis.label.set_color(col_strain_pred)
     ax3.set_facecolor("orange")
     ax3.patch.set_alpha(alpha)
     ax3.plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax3.plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax3.set_xlabel(x3, fontsize=axis_font)
     if y1:
      ax3.set_ylabel(y3_1, fontsize=axis_font)
      
     
     ax03 = ax3.twinx()
     ax03.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax03.yaxis.label.set_color(col_stress_pred)
     if y2:
      ax03.set_ylabel(y3_2, fontsize=axis_font)
     ax03.plot(x_pred_stress, y_pred_stress, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax03.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
    
     y_pred_strain = [x for _,x in sorted(zip(data_list[26],data_list[27]))] 
     x_pred_strain = sorted(data_list[26])
     y_tar_strain = [x for _,x in sorted(zip(data_list[24],data_list[25]))]     
     x_tar_strain = sorted(data_list[24])

     y_pred_stress = [x for _,x in sorted(zip(data_list[30],data_list[31]))] 
     x_pred_stress = sorted(data_list[30])
     y_tar_stress = [x for _,x in sorted(zip(data_list[28],data_list[29]))]     
     x_tar_stress = sorted(data_list[28])

     ax4.tick_params(axis='y', labelcolor=col_strain_pred, labelsize= tick_font)
     ax4.yaxis.label.set_color(col_strain_pred)
     # ax3.set_facecolor("orange")
     # ax3.patch.set_alpha(alpha)
     ax4.plot(x_pred_strain, y_pred_strain, label="Prediction", color=col_strain_pred,linestyle="solid", linewidth=width)   
     ax4.plot(x_tar_strain,y_tar_strain, label="Target", color=col_strain_tar,linestyle="dotted", linewidth=width)   
    
     ax4.set_xlabel(x4, fontsize=axis_font)
     if y1:
      ax4.set_ylabel(y4_1, fontsize=axis_font)
      
     
     ax04 = ax4.twinx()
     ax04.tick_params(axis='y', labelcolor=col_stress_pred, labelsize= tick_font)
     ax04.yaxis.label.set_color(col_stress_pred)
     if y2:
      ax04.set_ylabel(y4_2, fontsize=axis_font)
     ax04.plot(x_pred_stress, y_pred_stress, label="Prediction", color=col_stress_pred,linestyle="solid", linewidth=width)   
     ax04.plot(x_tar_stress,y_tar_stress, label="Target", color=col_stress_tar,linestyle="dotted", linewidth=width)   
    
     return


def plot_multi(data_1,data_2,data_4,labels_1,labels_2,labels_4,path):
     
     fig = plt.figure(layout='constrained', figsize=(36, 40))
     subfigs = fig.subfigures(1, 3, width_ratios = (1,0.90, 1))
     
     
     plot_curves(data_list=data_1, labels=labels_1, fig=subfigs[0],y1=True,y2=False)
     plot_curves(data_list=data_2, labels=labels_2, fig=subfigs[1],y1=False,y2=False)     
     plot_curves(data_list=data_4, labels=labels_4, fig=subfigs[2],y1=False, y2=True)
     
     fig.savefig(path+'figure_3.jpg', bbox_inches='tight',dpi=300)
     plt.close(fig)  
     return


def plot_distribution(tar,pred,interval,labels,path):
      title = labels[0]
      x = labels[1]
      y = labels[2]
      save = labels[3]
      fig,ax = plt.subplots()
      sns.distplot(tar, hist = False, kde = True,
                 kde_kws = {'shade':True,'linewidth': 3,"clip":interval},
                 hist_kws={"range":interval},label = "Target")
      sns.distplot(pred, hist = False, kde = True,
                 kde_kws = {'shade':True,'linewidth': 3,"clip":interval},
                 hist_kws={"range":interval},label = "Prediction")

      plt.title(label=title)
      plt.xlabel(x)
      plt.ylabel(y)
      
      plt.legend() 
      plt.savefig(path+save+'.png', bbox_inches='tight',dpi=300)
      plt.close()

def model_evaluation(model,test_input,test_target,epoch,volume,path,intervals_list,r):
    
    if epoch<10:
     checkpoint_path = "./tmp/model-000"+ str(epoch)+ ".h5"
    elif epoch<100 and epoch>=10:    
     checkpoint_path = "./tmp/model-00"+ str(epoch)+ ".h5"
    elif epoch<1000 and epoch>=100:    
     checkpoint_path = "./tmp/model-0"+ str(epoch)+ ".h5"
    else:    
     checkpoint_path = "./tmp/model-"+ str(epoch)+ ".h5"
   
    n_batches = int(test_input.shape[0])
    model.built = True
    model.load_weights(checkpoint_path)

    test_input = tf.cast(test_input,dtype=tf.float32)
    test_target = tf.cast(test_target,dtype=tf.float32)
    
    
    global_e_x_loss = 0
    global_e_yz_loss = 0
    global_s_x_loss = 0
    
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
        pred = model.NN.predict(test_input[i:i+1])
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
        
        ind_np = tf.where(tf.equal(test_input[i,:,:,-5],1))
        ind_intf = tf.where(tf.equal(test_input[i,:,:,-4],1))
        ind_bulk = tf.where(tf.equal(test_input[i,:,:,-3],1))
        
        
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


        # pred_global_s_x = tf.reduce_sum(tf.gather_nd(pred_s_x[:,:],ind_sx)).numpy()/(V_box*10000)
        # global_s_x = tf.reduce_sum(tf.gather_nd(test_target[i,:,:,2],ind_sx)).numpy()/(V_box*10000)
        
        pred_global_s_x = (s_x_int_pred*V_int+s_x_bulk_pred*V_bulk+s_x_np_pred*V_np)/V_box
        global_s_x = (s_x_int_tar*V_int+s_x_bulk_tar*V_bulk+s_x_np_tar*V_np)/V_box
        
        
        global_e_x_loss += tf.math.abs(tf.math.subtract(pred_global_e_x,global_e_x))
        global_e_yz_loss += tf.math.abs(tf.math.subtract(pred_global_e_yz,global_e_yz))       
        global_s_x_loss += tf.math.abs(tf.math.subtract(pred_global_s_x,global_s_x))

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
        
         


    
   
    global_e_x_loss /= n_batches
    global_e_yz_loss /= n_batches
    global_s_x_loss /= n_batches

    global_ex_bulk_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_e_x_bulk,tar_global_e_x_bulk)) )    
    global_ex_intp_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_e_x_intp,tar_global_e_x_intp)) )
    global_ex_np_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_e_x_np,tar_global_e_x_np)) )
    
    global_eyz_bulk_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_e_yz_bulk,tar_global_e_yz_bulk)) )    
    global_eyz_intp_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_e_yz_intp,tar_global_e_yz_intp)) )
    global_eyz_np_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_e_yz_np,tar_global_e_yz_np)) )
     
    global_sx_bulk_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_s_x_bulk,tar_global_s_x_bulk)) )    
    global_sx_intp_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_s_x_intp,tar_global_s_x_intp)) )
    global_sx_np_loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(pred_global_s_x_np,tar_global_s_x_np)) )

    
    min_s_bulk = np.min([pred_global_s_x_bulk,tar_global_s_x_bulk])
    min_s_int = np.min([pred_global_s_x_intp,tar_global_s_x_intp])
    min_s_np = np.min([pred_global_s_x_np,tar_global_s_x_np])
    min_s = np.min([pred_global_s_x_list,tar_global_s_x_list])
    
    tar_global_s_x_bulk -= min_s_bulk
    pred_global_s_x_bulk -= min_s_bulk
    tar_global_s_x_intp -= min_s_int
    pred_global_s_x_intp -= min_s_int
    tar_global_s_x_np -= min_s_np
    pred_global_s_x_np -= min_s_np
    tar_global_s_x_list -= min_s
    pred_global_s_x_list -= min_s



    
    print(float(volume),float(global_e_x_loss.numpy()),float(global_e_yz_loss.numpy()),float(global_s_x_loss.numpy()),float(global_ex_bulk_loss.numpy()),float(global_ex_intp_loss.numpy()),float(global_ex_np_loss.numpy()),float(global_eyz_bulk_loss.numpy()),float(global_eyz_intp_loss.numpy()),float(global_eyz_np_loss.numpy()),float(global_sx_bulk_loss.numpy()),float(global_sx_intp_loss.numpy()),float(global_sx_np_loss.numpy()),file=open("./mae.dat","a+"))
    print(float(volume),r2_score(tar_global_e_x_list,pred_global_e_x_list),r2_score(tar_global_e_yz_list,pred_global_e_yz_list),r2_score(tar_global_s_x_list,pred_global_s_x_list),r2_score(tar_global_e_x_bulk,pred_global_e_x_bulk),r2_score(tar_global_e_x_intp,pred_global_e_x_intp),r2_score(tar_global_e_x_np,pred_global_e_x_np),r2_score(tar_global_e_yz_bulk,pred_global_e_yz_bulk),r2_score(tar_global_e_yz_intp,pred_global_e_yz_intp),r2_score(tar_global_e_yz_np,pred_global_e_yz_np),r2_score(tar_global_s_x_bulk,pred_global_s_x_bulk),r2_score(tar_global_s_x_intp,pred_global_s_x_intp),r2_score(tar_global_s_x_np,pred_global_s_x_np),file=open("./r2.dat","a+"))
    
    np_list =[tar_global_e_x_np,tar_global_e_yz_np,pred_global_e_x_np,pred_global_e_yz_np,tar_global_e_x_np,tar_global_s_x_np,pred_global_e_x_np,pred_global_s_x_np
               ,tar_global_e_x_intp,tar_global_e_yz_intp,pred_global_e_x_intp,pred_global_e_yz_intp,tar_global_e_x_intp,tar_global_s_x_intp,pred_global_e_x_intp,pred_global_s_x_intp,
               tar_global_e_x_bulk,tar_global_e_yz_bulk,pred_global_e_x_bulk,pred_global_e_yz_bulk,tar_global_e_x_bulk,tar_global_s_x_bulk,pred_global_e_x_bulk,pred_global_s_x_bulk,
               tar_global_e_x_list,tar_global_e_yz_list,pred_global_e_x_list,pred_global_e_yz_list,tar_global_e_x_list,tar_global_s_x_list,pred_global_e_x_list,pred_global_s_x_list]   
    np_labels = [r"$\mathbf{{\epsilon}_{x}^{NP}}$",r"$\mathbf{{\epsilon}_{lat}^{NP}}$",r"$\mathbf{{\sigma}_{x}^{NP}}$ (GPa)",
                  r"$\mathbf{{\epsilon}_{x}^{I}}$",r"$\mathbf{{\epsilon}_{lat}^{I}}$",r"$\mathbf{{\sigma}_{x}^{I}}$ (GPa)",
                  r"$\mathbf{{\epsilon}_{x}^{M}}$",r"$\mathbf{{\epsilon}_{lat}^{M}}$",r"$\mathbf{{\sigma}_{x}^{M}}$ (GPa)",
                  r"$\mathbf{\epsilon_{x}^{global}}$",r"$\mathbf{\epsilon_{lat}^{global}}$",r"$\mathbf{\sigma_{x}^{global}}$ (GPa)",volume]
       
    
    print(total_time/n_batches)    
    return  np_list, np_labels   






v1_int_list = [-0.122815,0.773552,-0.150335,0.102609,-9686070.,11525500]
v2_int_list = [-0.118117,0.840645,-0.155665,0.085916,-10585700.,10702500]
# v3_int_list = [-0.153601,0.466508,-0.12719,0.091762,-7382320.0,9037000.0]
v4_int_list = [-0.115729,0.467162,-0.111824,0.094069,-9922250.0,8860930.0]
v5_int_list = [-0.111688, 0.423639,-0.107095,0.061978,-6757250.0,7242390.0]


v2 = 12.7
v1 = 16.1
# v3 = 1.9
v4 = 4.5
v5 = 7.6

r2 = 58
r1 = 52
r4 = 84
r5 = 69


epoch_list = [1500]
for i in epoch_list:
 with open(PATH+'/test_target_1.pkl','rb') as f:
    test_target_1 = pickle.load(f)
    print(test_target_1.shape)


 with open(PATH+'/test_input_1.pkl','rb') as f:
    test_input_1 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(test_input_1.shape) 



 with open(PATH+'/test_target_2.pkl','rb') as f:
    test_target_2 = pickle.load(f)
    print(test_target_2.shape)


 with open(PATH+'/test_input_2.pkl','rb') as f:
    test_input_2 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]    
    print(test_input_2.shape) 
    


 with open(PATH+'/test_target_4.pkl','rb') as f:
    test_target_4 = pickle.load(f)
    print(test_target_4.shape)


 with open(PATH+'/test_input_4.pkl','rb') as f:
    test_input_4 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(test_input_4.shape) 


 with open(PATH+'/test_target_5.pkl','rb') as f:
    test_target_5 = pickle.load(f)
    print(test_target_5.shape)


 with open(PATH+'/test_input_5.pkl','rb') as f:
    test_input_5 = pickle.load(f)[:,:,:,[0,1,2,3,6,7,8,9,10]]
    print(test_input_5.shape) 
 epoch = i
 PATH_2 = "./plots_12.7_"+str(i)+"/"
 PATH_1 = "./plots_16.1_"+str(i)+"/"
 PATH_4 = "./plots_4.5_"+str(i)+"/"
 PATH_5 = "./plots_7.6_"+str(i)+"/"
 PATH_7 = "./plots_"+str(i)+"/"

 data_1,labels_1 = model_evaluation(model,test_input_1,test_target_1,epoch,v1,PATH_1,v1_int_list,r1)   
 data_2,labels_2 = model_evaluation(model,test_input_2,test_target_2,epoch,v2,PATH_2,v2_int_list,r2)
 data_4,labels_4 = model_evaluation(model,test_input_4,test_target_4,epoch,v4,PATH_4,v4_int_list,r4)
 data_5,labels_5 = model_evaluation(model,test_input_5,test_target_5,epoch,v5,PATH_5,v5_int_list,r5)
 
 plot_multi(data_1, data_2, data_4, labels_1, labels_2, labels_4, PATH_7)
 plot_curves_5(data_1,labels_1, PATH_7)
 plot_curves_5(data_2,labels_2, PATH_7)
 plot_curves_5(data_4,labels_4, PATH_7)
 plot_curves_5(data_5,labels_5, PATH_7)
 plot_multi_scatters(data_1, labels_1, PATH_7)
 plot_multi_scatters(data_2, labels_2, PATH_7)
 plot_multi_scatters(data_4, labels_4, PATH_7)
 plot_multi_scatters(data_5, labels_5, PATH_7)
