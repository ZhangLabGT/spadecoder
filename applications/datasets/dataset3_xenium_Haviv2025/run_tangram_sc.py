import tangram as tg
import scanpy as sc
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
# from evaluations import *


# from multiprocessing import Process
# from multiprocessing import Semaphore

import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder.code import *


sc.settings.verbosity = 0


# adata_scrna_path = '../data/scrnaseq_ref.h5ad' #'/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/data/final_filtered.h5ad'
adata_scrna_path = '../data/scrna_ref_norm1knolog.h5ad' # '../data/scrnaseq_ref.h5ad' #'/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/data/final_filtered.h5ad'

# types_of_spots = ['linear']
# num_simulations = 1 # 3
num_tissueslices = 10 # 10

result_metric = ['orig_rmse',   'avg_corr_pe','avg_jsd'] # ['orig_rmse', 'new_rmse', 'ari', 'purity','avg_corr_sp','avg_corr_pe','avg_jsd']
scrna_cluster_key = "cell_type"

resdir = '../results/' # '/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/results/'

simdir = resdir + 'simulations/slice_warps/'

N_base = 50
nneigh = 10 
nswaps_nbd_base = 2


# par_dict = {'N':[ 5, 10, 50,  100,  200], 
#             'nswaps_nbd':[2, 5, 10, 20, 50, 100]}

par_dict = {'N':[25,  75]}# [ 5, 10, 50,  100,  200]
            #'nswaps_nbd':[2, 5, 10, 20, 50, 100]}

suffix = '_norm1knolog'


def tangram_run(nneigh, adata_sc, **kwargs):


    N_curr = kwargs.get('N', N_base)
    nswaps_nbd_curr = kwargs.get('nswaps_nbd', nswaps_nbd_base)
    
    save_sim_slices = '../results/simulations/pickles/' + 'multi_slice_simulated_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + suffix + '.pickle' # '_old.pickle'
    
    with open(save_sim_slices, 'rb') as handle:
                adata_spot_swap = pickle.load(handle)

    
    deconv_st1 = {}
    results_df = {}

    print(save_sim_slices)

    # for entry in types_of_spots: # linear, polar, gauss warp 
    #     for entry2 in range(num_simulations): # number of times each simulation is run
    key_name = 'linear_0' # entry + '_' + str(entry2)

    deconv_st1[key_name] = {}
    results_df[key_name] = {}

    real_samples = list(adata_spot_swap[key_name][0].keys())

    time_st1 = pd.DataFrame(0.0,index=real_samples, columns=list(range(num_tissueslices+1)))
        
    for entry3 in real_samples: # iterate over real input samples or slices  
        
        results_df[key_name][entry3] = pd.DataFrame(index=[metric_entry + '_avg' for metric_entry in result_metric])
        
        deconv_st1[key_name][entry3] = {}
        
        for entry0 in range(num_tissueslices+1): # iterate over simulated tissue slices for each input slice
            
            
            adata_spot = adata_spot_swap[key_name][entry0][entry3].copy()

            start_time = time.time()

            tg.pp_adatas(adata_sc, adata_spot)

            ad_map = tg.map_cells_to_space(
                adata_sc,
                adata_spot,
                mode='cells',verbose=False,device="cuda:0")
                # cluster_label=scrna_cluster_key)
            
            tg.project_cell_annotations(ad_map, adata_spot, annotation=scrna_cluster_key)

            celltype_density = adata_spot.obsm['tangram_ct_pred']
            celltype_density = (celltype_density.T/celltype_density.sum(axis=1))

            end_time = time.time()
            time_st1.loc[entry3,entry0] = float(f"{(end_time - start_time):.3f}")
            print(f"Runtime: {time_st1.loc[entry3,entry0]:.3f} seconds")

            deconv_st1[key_name][entry3][entry0] = celltype_density # [adata_spot_swap[key_name][entry0][entry3].obs.columns].T
            
            # print(deconv_st1[key_name][entry3][entry0])

            # print(adata_spot_swap[key_name][entry0][entry3].obs)

            results_df[key_name][entry3][entry0],_ = eval_deconv3(adata_spot_swap[key_name][entry0][entry3].obs,deconv_st1[key_name][entry3][entry0])
                
            # celltype_density.to_csv(op_file,sep='\t')
                        
    
    write_slice1 = simdir + 'deconv_st1_'  + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr)   + '_tangramsc.pickle'
    with open(write_slice1, 'wb') as handle:
            pickle.dump(deconv_st1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    
    
    metrics_slice1  = simdir + 'metrics_st1_'  + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + '_tangramsc.pickle'
    with open(metrics_slice1, 'wb') as handle:
            pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    metrics_slice1  = simdir + 'runtimes_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + '_tangramsc.csv'
    time_st1.to_csv(metrics_slice1)

    # sema.release() 


if __name__ == "__main__":
    # N_range =  [5, 10, 20, 50, 75, 100, 150] #, 250, 400, 500] # []  # [] #  # [50] # #  #  #  #   #  #  # [250, 400, 500]#[5, 10] #,  50, 100, 250, 500] # [5, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # linear_slope_variance_range = [0.1] # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # linear_intercept_variance_range = [0.1] # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # numcells_to_swap_range = [2] # in [0, 2, 4, 8, 16, 32, 64, 128, 256]:
    # # num_metacells = [750, 1000,  2000,  3000,   5000, 15000] # 
    # num_metacells = [50, 100, 250, 500, 750, 1000,  2000,  3000,   5000] # , 15000]


    adata_sc = sc.read(adata_scrna_path)

    # args = []
    # procs = []

    for par, par_values in par_dict.items():
        for par_value in par_values:
        # for linear_slope_variance in linear_slope_variance_range: #[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        #     for linear_intercept_variance in linear_intercept_variance_range: # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        #         for numcells_to_swap in numcells_to_swap_range: # [0, 2, 4, 8, 16, 32, 64, 128, 256]:
            tangram_run( nneigh, adata_sc,**{par:par_value})
    # for N in N_range:
    #     for linear_slope_variance in linear_slope_variance_range: #[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    #         for linear_intercept_variance in linear_intercept_variance_range: # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    #             for numcells_to_swap in numcells_to_swap_range: # [0, 2, 4, 8, 16, 32, 64, 128, 256]:
    #                 args.append((N, linear_slope_variance,linear_intercept_variance, numcells_to_swap, adata_sc))

    