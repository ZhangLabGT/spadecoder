import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder.code import *


import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
import gc
import glob
import pickle
from scipy.sparse import csr_matrix


# from multiprocessing import Process
# from multiprocessing import Semaphore



# this line forces theano to use the GPU and should go before importing cell2location
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=float32,force_device=True'
# if using the CPU uncomment this:
# os.environ["THEANO_FLAGS"] = 'device=cpu,floatX=float32,openmp=True,force_device=True'

import cell2location
from cell2location.models import RegressionModel



# silence scanpy that prints a lot of warnings
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

######## variables begin ################

# need raw counts 
adata_scrna_path = "../data/scrnaseq_raw.h5ad" #'/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/data/final_filtered.h5ad'
# types_of_spots = ['linear']
# num_simulations = 1 #3
num_tissueslices = 10 #10
result_metric = ['orig_rmse',  'avg_corr_pe','avg_jsd'] # ['orig_rmse', 'new_rmse', 'ari', 'purity','avg_corr_sp','avg_corr_pe','avg_jsd']
scrna_cluster_key = "Cell class (determined from clustering of all cells)"
resdir = '../results/' # '/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/results/'
simdir = resdir + 'simulations/slice_warps/'
################### variables end 


# #### cell2loc parameters ###### 
# detection_alpha=200 # recommended is trying 20 and 200
# ######################


N_base = 50
nneigh = 10 
nswaps_nbd_base = 2


# par_dict = {'N':[ 10, 50,  100,  200],  # 5
#             'nswaps_nbd':[2, 5, 10, 20, 50, 100]}

par_dict = { 'N':[100]}# 50, 25, 10,  75, 100]#  # 5 # [ 5, 10, 50,  100,  200]
            # 'nswaps_nbd':[5, 10, 20]} # 2, 5, 10, 20, 50, 100]}



suffix = '_normnolog_nov2024_1k'


def cell2location_run(nneigh, inf_aver, **kwargs):

    # find shared genes and subset both anndata and reference signatures
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
            
            print(entry3, entry0)
            
            adata_spot = adata_spot_swap[key_name][entry0][entry3].copy()
            #med_cells_per_spot = adata_spot.obs.sum(axis=1).median().astype(int)
            # make pseudo raw counts 
            expr = (adata_spot.X * 100).astype(int).astype(float) # adata_spot.obs.sum(axis=1).values[:,np.newaxis]
            adata_spot.X = csr_matrix(expr)
            
            
            intersect = np.intersect1d(adata_spot.var_names, inf_aver.index)
            adata_spot = adata_spot[:, intersect].copy()
            inf_aver = inf_aver.loc[intersect, :]# .copy()
            

            start_time = time.time()
            # prepare anndata for cell2location model
            cell2location.models.Cell2location.setup_anndata(adata_spot)

            # gc.collect()
            
            mod = cell2location.models.Cell2location(adata_spot, cell_state_df=inf_aver) 
                                                        # the expected average cell abundance: tissue-dependent
                                                    # hyper-prior which can be estimated from paired histology:
                                                       #  N_cells_per_location=med_cells_per_spot,
                                                        # hyperparameter controlling normalisation of
                                                    # within-experiment variation in RNA detection (using default here):
                                                        # detection_alpha=detection_alpha)
            

            mod.train(use_gpu=True)
            

            adata_spot = mod.export_posterior(adata_spot,sample_kwargs={ 'batch_size': mod.adata.n_obs}) # , sample_kwargs={'use_gpu':True}
            
            result3 = adata_spot.obsm['means_cell_abundance_w_sf']
            sum_result_3 = result3.sum(axis=1)
            result3_percent = result3.div(result3.assign(total=sum_result_3)['total'], axis='index') # this is cells by cell-type
            # input to eval_deconv3 is 
            # 1. ground truth -> cell by type df
            # 2. deconvolution results -> cell-type by cell
            
            end_time = time.time()
            time_st1.loc[entry3,entry0] = float(f"{(end_time - start_time):.3f}") 
            print(f"Runtime: {time_st1.loc[entry3,entry0]:.3f} seconds")
             
            deconv_st1[key_name][entry3][entry0] = result3_percent.T.copy()
            
            deconv_st1[key_name][entry3][entry0].index =  [entry_idx.split('_')[-1] for entry_idx in list(deconv_st1[key_name][entry3][entry0].index)]
        
            results_df[key_name][entry3][entry0],_ = eval_deconv3(adata_spot_swap[key_name][entry0][entry3].obs,deconv_st1[key_name][entry3][entry0])



    write_slice1 = simdir + 'deconv_st1_'  + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + '_cell2location.pickle'
    with open(write_slice1, 'wb') as handle:
            pickle.dump(deconv_st1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    
    
    metrics_slice1  = simdir + 'metrics_st1_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + '_cell2location.pickle'
    with open(metrics_slice1, 'wb') as handle:
            pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_slice1  = simdir + 'runtimes_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + '_cell2location.csv'
    time_st1.to_csv(metrics_slice1)
              
    # sema.release() 
                    

if __name__ == "__main__":
    # N_range =   [5, 10, 20, 50, 75, 100, 150, 250, 400, 500] # # [5], []  # [] #  # [50] # #  #  #  #   #  #  # [250, 400, 500]#[5, 10] #,  50, 100, 250, 500] # [5, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # N_range.reverse()
    # linear_slope_variance_range = [0.1] # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # linear_intercept_variance_range = [0.1] # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # numcells_to_swap_range = [2] # in [0, 2, 4, 8, 16, 32, 64, 128, 256]:
    # # num_metacells = [750, 1000,  2000,  3000,   5000, 15000] # 
    # num_metacells = [50, 100, 250, 500, 750, 1000,  2000,  3000,   5000] # , 15000]

    if os.path.exists("../data/cell2loc_reference.csv"):
        inf_aver = pd.read_csv("../data/cell2loc_reference.csv",index_col=0)
         
    else:
        ###########
        # read scRNA
        adata_sc = sc.read(adata_scrna_path)
        ##########

        #############
        # prep scRNA 
        cell2location.models.RegressionModel.setup_anndata(adata=adata_sc,labels_key=scrna_cluster_key)
        mod = RegressionModel(adata_sc)
        # Use all data for training (validation not implemented yet, train_size=1)
        mod.train()

        # plot ELBO loss history during training, removing first 20 epochs from the plot
        # mod.plot_history(20)
        # In this section, we export the estimated cell abundance (summary of the posterior distribution).
        adata_sc = mod.export_posterior(
            adata_sc
        )

        if 'means_per_cluster_mu_fg' in adata_sc.varm.keys():
            inf_aver = adata_sc.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_sc.uns['mod']['factor_names']]].copy()
        else:
            inf_aver = adata_sc.var[[f'means_per_cluster_mu_fg_{i}'
                                        for i in adata_sc.uns['mod']['factor_names']]].copy()
        inf_aver.columns = adata_sc.uns['mod']['factor_names']

        inf_aver.to_csv("../data/cell2loc_reference.csv")
        ####################

    # args = []
    # procs = []

    for par, par_values in par_dict.items():
        for par_value in par_values:
        # for linear_slope_variance in linear_slope_variance_range: #[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        #     for linear_intercept_variance in linear_intercept_variance_range: # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        #         for numcells_to_swap in numcells_to_swap_range: # [0, 2, 4, 8, 16, 32, 64, 128, 256]:
                    # args.append((N, linear_slope_variance,linear_intercept_variance, numcells_to_swap, inf_aver))
            cell2location_run( nneigh, inf_aver, **{par:par_value})
            

    

    # concurrency = 10
    # sema = Semaphore(concurrency)
    # for arg in args:
    #     # print(name)
    #     sema.acquire()
    #     proc = Process(target=cell2location_run, args=arg)
    #     procs.append(proc)
    #     proc.start()
   
   
    # for proc in procs:
    #     proc.join()

    