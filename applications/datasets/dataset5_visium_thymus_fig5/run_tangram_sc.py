import tangram as tg
import scanpy as sc
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle


import sys
package_path = "/localscratch/mlobo6/spadecoder/datasets"
if package_path not in sys.path:
    sys.path.append(package_path)
from spadecoder.code import *


sc.settings.verbosity = 0


adata_scrna_path = '../data/scRNA_norm.h5ad' 


resdir = '../results/' 

simdir = resdir + 'simulations/slice_warps/'



ctype_list = ['cell_type_level_0_episub'] # , 

def tangram_run(adata_sc, scrna_cluster_key ):

    adata_spa_path = '../data/spatial_norm.pickle'
    with open(adata_spa_path, 'rb') as handle:
        adata_spa = pickle.load(handle)

    deconv_st1 = {}
    cell_wts = {}
    
    real_samples = list(adata_spa.keys())
    # slice_samples =  list(adata_spa[real_samples[0]].keys())

    time_st1 = pd.DataFrame(0.0,index=real_samples, columns=[0])
    
    adata_sc_orig = adata_sc.copy()

    for real_curr in real_samples: # iterate over real input samples or slices  
        
        deconv_st1[real_curr] = {}
        cell_wts[real_curr] = {}
        
        

        adata_spot = adata_spa[real_curr].copy()
        adata_spot = adata_spot[:,adata_sc_orig.var.index]
        

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
        time_st1.loc[real_curr,0] = float(f"{(end_time - start_time):.3f}")
        print(f"Runtime: {time_st1.loc[real_curr,0]:.3f} seconds")

        deconv_st1[real_curr][0] = celltype_density 
        
        cell_wts[real_curr][0] =  ad_map.to_df()

        
                        
    
    write_slice1 = simdir + 'deconv_st1_'  + 'anno_type_' + scrna_cluster_key + '_tangramsc.pickle'
    with open(write_slice1, 'wb') as handle:
        pickle.dump(deconv_st1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    metrics_slice1  = simdir + 'runtimes_' + 'anno_type_' + scrna_cluster_key + '_tangramsc.csv'
    time_st1.to_csv(metrics_slice1)

    write_slice1 = simdir + 'cellwts_' + 'anno_type_' + scrna_cluster_key +   '_tangramsc_for_spaloc.pickle'
    with open(write_slice1, 'wb') as handle:
        pickle.dump(cell_wts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # sema.release() 
    return


if __name__ == "__main__":
    
    adata_sc = sc.read(adata_scrna_path)

    for scrna_cluster_key in ctype_list:
        torch.cuda.empty_cache()
        tangram_run( adata_sc,scrna_cluster_key)