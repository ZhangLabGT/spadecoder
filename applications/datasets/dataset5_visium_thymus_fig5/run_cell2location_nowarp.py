import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets/"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder.evaluations import *



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
adata_scrna_path = "../data/scRNA_raw.h5ad" #'/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/data/final_filtered.h5ad'

resdir = '../results/' # '/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/results/'
simdir = resdir + 'simulations/slice_warps/'
################### variables end 


#### cell2loc parameters ###### 
# detection_alpha=200 # recommended is trying 20 and 200
######################



ctype_list = ['cell_type_level_0_episub']

def cell2location_run( inf_aver, scrna_cluster_key):

    # find shared genes and subset both anndata and reference signatures
    # N_curr = kwargs.get('N', N_base)
    # nswaps_nbd_curr = kwargs.get('nswaps_nbd', nswaps_nbd_base)

    # save_sim_slices = '../results/simulations/pickles/' + 'multi_slice_simulated_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + suffix + '.pickle' # '_old.pickle'
    
    adata_spa_path = '../data/spatial_raw.pickle'
    with open(adata_spa_path, 'rb') as handle:
        adata_spa = pickle.load(handle)

    
    deconv_st1 = {}
    # results_df = {}


    real_samples = list(adata_spa.keys())
    # slice_samples =  list(adata_spa[real_samples[0]].keys())

    time_st1 = pd.DataFrame(0.0,index=real_samples, columns=[0])
    

    
    for real_curr in real_samples: # iterate over real input samples or slices  

        # results_df[entry3] = pd.DataFrame(index=[metric_entry + '_avg' for metric_entry in result_metric])
        
        deconv_st1[real_curr] = {}
        
        # for slice_curr in slice_samples: # iterate over simulated tissue slices for each input slice
            
        # print(real_curr, slice_curr)
        adata_spot = adata_spa[real_curr].copy()
        # med_cells_per_spot = adata_spot.obs.sum(axis=1).median().astype(int)
        # make pseudo raw counts 
        # expr = (adata_spot.X * 100).astype(int).astype(float) # adata_spot.obs.sum(axis=1).values[:,np.newaxis]
        # adata_spot.X = csr_matrix(expr)
        
        intersect = np.intersect1d(adata_spot.var_names, inf_aver.index)
        adata_spot = adata_spot[:, intersect].copy()
        inf_aver = inf_aver.loc[intersect, :]# .copy()

        
        # prepare anndata for cell2location model
        cell2location.models.Cell2location.setup_anndata(adata_spot)

        # gc.collect()
        start_time = time.time()
        mod = cell2location.models.Cell2location(adata_spot, cell_state_df=inf_aver) 

        mod.train()

        adata_spot = mod.export_posterior(adata_spot,sample_kwargs={ 'batch_size': mod.adata.n_obs})


        result3 = adata_spot.obsm['means_cell_abundance_w_sf']
        sum_result_3 = result3.sum(axis=1)
        result3_percent = result3.div(result3.assign(total=sum_result_3)['total'], axis='index') # this is cells by cell-type
        # input to eval_deconv3 is 
        # 1. ground truth -> cell by type df
        # 2. deconvolution results -> cell-type by cell
        end_time = time.time()
        time_st1.loc[real_curr,0] = float(f"{(end_time - start_time):.3f}") 
        print(f"Runtime: {time_st1.loc[real_curr,0]:.3f} seconds")
        
        deconv_st1[real_curr] = result3_percent.T.copy()

        # print(deconv_st1[key_name][entry3][entry0].head())

        # deconv_st1[key_name][entry3][entry0].index =  [entry_idx.split('_')[-1] for entry_idx in list(deconv_st1[key_name][entry3][entry0].index)]

        # print(deconv_st1[key_name][entry3][entry0].head())

        deconv_st1[real_curr].index = [entry_idx.replace('meanscell_abundance_w_sf_','') for entry_idx in list(deconv_st1[real_curr].index)]


        # results_df[key_name][real_curr][entry0],_ = eval_deconv3(adata_spot_swap[key_name][entry0][real_curr].obs,deconv_st1[key_name][real_curr][entry0])

        # print(adata_spot_swap[key_name][entry0][entry3].obs.head())

    write_slice1 = simdir + 'deconv_st1_'  + 'anno_type_' + scrna_cluster_key + '_cell2location.pickle'
    with open(write_slice1, 'wb') as handle:
        pickle.dump(deconv_st1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    
    
    # metrics_slice1  = simdir + 'metrics_st1_' + 'sptsz_' + str(N_curr)  + '_nneigh_' + str(nneigh) + '_nbdswaps_' + str(nswaps_nbd_curr) + '_cell2location.pickle'
    # with open(metrics_slice1, 'wb') as handle:
    #         pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_slice1  = simdir + 'runtimes_' + 'anno_type_' + scrna_cluster_key + '_cell2location.csv'
    time_st1.to_csv(metrics_slice1)
              
    # sema.release() 
                    

if __name__ == "__main__":
    
    for scrna_cluster_key in ctype_list:
        print("Running cell2location for scrna_cluster_key:", scrna_cluster_key)
        

        if os.path.exists("../data/cell2loc_reference_" +  'anno_type_' + scrna_cluster_key + ".csv"):
            inf_aver = pd.read_csv("../data/cell2loc_reference_" +  'anno_type_' + scrna_cluster_key + ".csv",index_col=0)
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

            inf_aver.to_csv("../data/cell2loc_reference_" +  'anno_type_' + scrna_cluster_key + ".csv")
    

        cell2location_run( inf_aver, scrna_cluster_key)
            

    