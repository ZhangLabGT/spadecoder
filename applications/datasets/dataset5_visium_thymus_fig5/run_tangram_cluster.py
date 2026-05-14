import tangram as tg
import scanpy as sc
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
# from evaluations import *


from multiprocessing import Process
from multiprocessing import Semaphore

import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder.code import *


sc.settings.verbosity = 0


# adata_scrna_path = '../data/scrnaseq_ref.h5ad' #'/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/data/final_filtered.h5ad'
adata_scrna_path = '../data/scRNA_norm.h5ad' # '../data/scrnaseq_ref.h5ad' #'/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/data/final_filtered.h5ad'


resdir = '../results/' # '/Users/macrinalobo/Documents/zhanglab/2024AprSpadecoder/dataset1_merfish_moffitt2018_50/results/'

simdir = resdir + 'simulations/slice_warps/'


ctype_list = ['cell_type_level_0_episub'] 


def tangram_run( adata_sc, scrna_cluster_key ):
    
    # N_curr = kwargs.get('N', N_base)
    # nswaps_nbd_curr = kwargs.get('nswaps_nbd', nswaps_nbd_base)

    adata_spa_path = '../data/spatial_norm.pickle'
    with open(adata_spa_path, 'rb') as handle:
        adata_spa = pickle.load(handle)

    
    deconv_st1 = {}
   

    real_samples = list(adata_spa.keys())
    time_st1 = pd.DataFrame(0.0,index=real_samples, columns=[0])
    

    adata_sc_orig = adata_sc.copy()

    
    for real_curr in  real_samples: # iterate over real input samples or slices  
        
       
        deconv_st1[real_curr] = {}
        
        

        adata_spot = adata_spa[real_curr].copy()
        adata_spot = adata_spot[:,adata_sc_orig.var.index]

        start_time = time.time()

        tg.pp_adatas(adata_sc, adata_spot)

        
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_spot,
            mode='clusters',verbose=False, device="cuda:0",
            cluster_label=scrna_cluster_key)
        
        tg.project_cell_annotations(ad_map, adata_spot, annotation=scrna_cluster_key)

        celltype_density = adata_spot.obsm['tangram_ct_pred']
        celltype_density = (celltype_density.T/celltype_density.sum(axis=1))

        end_time = time.time()
        time_st1.loc[real_curr,0] = float(f"{(end_time - start_time):.3f}")
        print(f"Runtime: {time_st1.loc[real_curr,0]:.3f} seconds")

        deconv_st1[real_curr][0] = celltype_density # [adata_spot_swap[key_name][entry0][entry3].obs.columns].T
        
       
                    
    
    write_slice1 = simdir + 'deconv_st1_'  + 'anno_type_' + scrna_cluster_key + '_tangram.pickle'
    with open(write_slice1, 'wb') as handle:
        pickle.dump(deconv_st1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    
    

    metrics_slice1  = simdir + 'runtimes_' + 'anno_type_' + scrna_cluster_key + '_tangram.csv'
    time_st1.to_csv(metrics_slice1)
       
    return
    # sema.release() 


if __name__ == "__main__":
    


    adata_sc = sc.read(adata_scrna_path)

    for scrna_cluster_key in ctype_list:


        tangram_run( adata_sc,scrna_cluster_key)
