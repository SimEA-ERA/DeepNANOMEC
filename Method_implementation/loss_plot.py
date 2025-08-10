# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 23:32:20 2021

@author: e.christofi
"""

import matplotlib.pyplot as plt
import pickle

with open('./losses.pkl','rb') as f:
    losses = pickle.load(f)

def make_plots(labels,data):
  x1_label = labels[0]
  x2_label = labels[1]
  x3_label = labels[2]

  loss_label1 = labels[3]  
  loss_label2 = labels[4]  
  loss_label3 = labels[5]  

  width = 3.     
  axis_font = 50
  tick_font = 28 
  fig = plt.figure(layout='constrained',figsize=(10,20))
  ax = fig.subplots(3,1,sharex=True)
  
  ax[0].tick_params(axis='y', labelsize= tick_font)
  ax[0].tick_params(axis='x', labelsize= tick_font)
  
  ax[0].plot(losses[loss_label1][:1500],color="red", linewidth = width)
  ax[0].plot(losses["val_"+loss_label1][:1500],color="blue",linestyle="dotted", linewidth = width)
  # ax[0].set_xlabel("Epoch", fontsize=axis_font)
  ax[0].set_ylabel(x1_label, fontsize=axis_font)
  ax[0].set_yscale("log")

  ax[1].tick_params(axis='y', labelsize= tick_font)
  ax[1].tick_params(axis='x', labelsize= tick_font)
  
  ax[1].plot(losses[loss_label2][:1500],color="red", linewidth = width)
  ax[1].plot(losses["val_"+loss_label2][:1500],color="blue",linestyle="dotted", linewidth = width)
  # ax[1].set_xlabel("Epoch", fontsize=axis_font)
  ax[1].set_ylabel(x2_label, fontsize=axis_font)
  ax[1].set_yscale("log")

  ax[2].tick_params(axis='y', labelsize= tick_font)
  ax[2].tick_params(axis='x', labelsize= tick_font)
  
  ax[2].plot(losses[loss_label3][:1500],color="red", linewidth = width)
  ax[2].plot(losses["val_"+loss_label3][:1500],color="blue",linestyle="dotted", linewidth = width)
  ax[2].set_xlabel("Epoch", fontsize=axis_font)
  ax[2].set_ylabel(x3_label, fontsize=axis_font)
  ax[2].set_yscale("log")
  
  
  fig.savefig("./losses.jpg", bbox_inches='tight',dpi=300)
  plt.close(fig) 
  # plt.show()
label_list = [r"$\mathcal{L}_{atom}$",r"$\mathcal{L}_{region}$",r"$\mathcal{L}_{atom}+\mathcal{L}_{region}$",
              "atom_loss","region_loss","loss"]
make_plots(label_list,losses)