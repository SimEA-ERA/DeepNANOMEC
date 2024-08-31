# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:47:14 2024

@author: e.christofi
"""
import numpy as np

labels = ["volume","global_e_x","global_e_lat","global_s_x","e_x_bulk","e_x_intp","e_x_np","e_lat_bulk","e_lat_intp","e_lat_np","s_x_bulk","s_x_intp","s_x_np"]
run_labels = [1000,1500,2000,2500,3000]
def calc_min_max_error(r2_file,raw,col,max_v,min_v,max_l,min_l,i,epoch):
    if r2_file[raw+(4*i),col] >= max_v :
        max_v = r2_file[raw+(4*i),col]
        max_l = epoch
    elif r2_file[raw+(4*i),col] < min_v :
        min_v = r2_file[raw+(4*i),col]
        min_l = epoch    
    return max_v,max_l,min_v,min_l   
        
def calc_min_max_runs(vol):
  if vol==16.1:
      raw=0
  elif vol==12.7:
      raw=1
  elif vol==4.5:
      raw=2
  if vol==7.6:
      raw=3

  min_g_ex = 10  
  min_g_elat = 10
  min_g_sx = 10
  min_ex_bulk = 10
  min_elat_bulk = 10
  min_sx_bulk = 10
  min_ex_intp = 10
  min_elat_intp = 10
  min_sx_intp = 10
  min_ex_np = 10
  min_elat_np = 10
  min_sx_np = 10

  max_g_ex = -1000 
  max_g_elat = -1000
  max_g_sx = -1000
  max_ex_bulk = -1000
  max_elat_bulk = -1000
  max_sx_bulk = -1000
  max_ex_intp = -1000
  max_elat_intp = -1000
  max_sx_intp = -1000
  max_ex_np = -1000
  max_elat_np = -1000
  max_sx_np = -1000
  
  max_g_ex_run = 0
  max_g_elat_run = 0
  max_g_sx_run = 0
  max_ex_bulk_run = 0
  max_ex_intp_run = 0
  max_ex_np_run = 0
  max_elat_bulk_run = 0
  max_elat_intp_run = 0
  max_elat_np_run = 0
  max_sx_bulk_run = 0
  max_sx_intp_run = 0
  max_sx_np_run = 0

  min_g_ex_run = 0
  min_g_elat_run = 0
  min_g_sx_run = 0
  min_ex_bulk_run = 0
  min_ex_intp_run = 0
  min_ex_np_run = 0
  min_elat_bulk_run = 0
  min_elat_intp_run = 0
  min_elat_np_run = 0
  min_sx_bulk_run = 0
  min_sx_intp_run = 0
  min_sx_np_run = 0

  for i in range(len(run_labels)):
    r2_file = np.loadtxt("./r2.dat")
    epoch = run_labels[i]
    max_g_ex,max_g_ex_run,min_g_ex,min_g_ex_run = calc_min_max_error(r2_file,raw,1,max_g_ex,min_g_ex,max_g_ex_run,min_g_ex_run,i,epoch)
    max_g_elat,max_g_elat_run,min_g_elat,min_g_elat_run =calc_min_max_error(r2_file,raw,2,max_g_elat,min_g_elat,max_g_elat_run,min_g_elat_run,i,epoch)
    max_g_sx,max_g_sx_run,min_g_sx,min_g_sx_run =calc_min_max_error(r2_file,raw,3,max_g_sx,min_g_sx,max_g_sx_run,min_g_sx_run,i,epoch)
    max_ex_bulk,max_ex_bulk_run,min_ex_bulk,min_ex_bulk_run = calc_min_max_error(r2_file,raw,4,max_ex_bulk,min_ex_bulk,max_ex_bulk_run,min_ex_bulk_run,i,epoch)
    max_ex_intp,max_ex_intp_run,min_ex_intp,min_ex_intp_run = calc_min_max_error(r2_file,raw,5,max_ex_intp,min_ex_intp,max_ex_intp_run,min_ex_intp_run,i,epoch)
    max_ex_np,max_ex_np_run,min_ex_np,min_ex_np_run = calc_min_max_error(r2_file,raw,6,max_ex_np,min_ex_np,max_ex_np_run,min_ex_np_run,i,epoch)
    max_elat_bulk,max_elat_bulk_run,min_elat_bulk,min_elat_bulk_run = calc_min_max_error(r2_file,raw,7,max_elat_bulk,min_elat_bulk,max_elat_bulk_run,min_elat_bulk_run,i,epoch)
    max_elat_intp,max_elat_intp_run,min_elat_intp,min_elat_intp_run = calc_min_max_error(r2_file,raw,8,max_elat_intp,min_elat_intp,max_elat_intp_run,min_elat_intp_run,i,epoch)
    max_elat_np,max_elat_np_run,min_elat_np,min_elat_np_run =calc_min_max_error(r2_file,raw,9,max_elat_np,min_elat_np,max_elat_np_run,min_elat_np_run,i,epoch)
    max_sx_bulk,max_sx_bulk_run,min_sx_bulk,min_sx_bulk_run =calc_min_max_error(r2_file,raw,10,max_sx_bulk,min_sx_bulk,max_sx_bulk_run,min_sx_bulk_run,i,epoch)
    max_sx_intp,max_sx_intp_run,min_sx_intp,min_sx_intp_run =calc_min_max_error(r2_file,raw,11,max_sx_intp,min_sx_intp,max_sx_intp_run,min_sx_intp_run,i,epoch)
    max_sx_np,max_sx_np_run,min_sx_np,min_sx_np_run =calc_min_max_error(r2_file,raw,12,max_sx_np,min_sx_np,max_sx_np_run,min_sx_np_run,i,epoch)
  
  print("Volume:",vol)
  print("Min global e_x:", min_g_ex, " Run:",min_g_ex_run)   
  print("Max global e_x:", max_g_ex, " Run:",max_g_ex_run)   
  print("Min global e_lat:", min_g_elat, " Run:",min_g_elat_run)   
  print("Max global e_lat:", max_g_elat, " Run:",max_g_elat_run)   
  print("Min global s_x:", min_g_sx, " Run:",min_g_sx_run)   
  print("Max global s_x:", max_g_sx, " Run:",max_g_sx_run)   
  print("Min e_x bulk:", min_ex_bulk, " Run:",min_ex_bulk_run)   
  print("Max e_x bulk:", max_ex_bulk, " Run:",max_ex_bulk_run)   
  print("Min e_x intp:", min_ex_intp, " Run:",min_ex_intp_run)   
  print("Max e_x intp:", max_ex_intp, " Run:",max_ex_intp_run)   
  print("Min e_x np:", min_ex_np, " Run:",min_ex_np_run)   
  print("Max e_x np:", max_ex_np, " Run:",max_ex_np_run)   
  print("Min e_lat bulk:", min_elat_bulk, " Run:",min_elat_bulk_run)   
  print("Max e_lat bulk:", max_elat_bulk, " Run:",max_elat_bulk_run)   
  print("Min e_lat intp:", min_elat_intp, " Run:",min_elat_intp_run)   
  print("Max e_lat intp:", max_elat_intp, " Run:",max_elat_intp_run)   
  print("Min e_lat np:", min_elat_np, " Run:",min_elat_np_run)   
  print("Max e_lat np:", max_elat_np, " Run:",max_elat_np_run)   
  print("Min s_x bulk:", min_sx_bulk, " Run:",min_sx_bulk_run)   
  print("Max s_x bulk:", max_sx_bulk, " Run:",max_sx_bulk_run)   
  print("Min s_x intp:", min_sx_intp, " Run:",min_sx_intp_run)   
  print("Max s_x intp:", max_sx_intp, " Run:",max_sx_intp_run)   
  print("Min s_x np:", min_sx_np, " Run:",min_sx_np_run)   
  print("Max s_x np:", max_sx_np, " Run:",max_sx_np_run)   
  print("")
    
  
calc_min_max_runs(vol=16.1)
calc_min_max_runs(vol=12.7)
calc_min_max_runs(vol=4.5)
calc_min_max_runs(vol=7.6)
    