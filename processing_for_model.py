from .importing_modules import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_gene_expr_from_scrna_h5ad(filepath, cluster_key, layer='raw', 
                                  genes_to_use='all',scrna_var_key='index'):
    # input is the filepath to the processed h5ad file 
    
#     # STEPs:
#     1. read the object 
#     2. get the cluster average gene expression
    
#     # output
#     a. h5ad scrna object
#     b. cluster average gene expression
#     if layer =='raw':
#         getX = lambda x: x.raw.to_adata().X
#     elif layer is not None:
#             getX = lambda x: x.layers[layer]
#     else:
#         getX = lambda x: x.X
# #     if gene_symbols is not None:
#         new_idx = adata.var[idx]
#     else:
#         new_idx = adata.var_names

    adata_scrna = sc.read(filepath)
    
    
    if layer == 'raw':
        adata = adata_scrna.raw.to_adata()
    else:
        adata = adata_scrna.copy()

    del adata_scrna

    if scrna_var_key != 'index':
        var_df = sc.get.var_df(adata,keys=list(adata.var.columns))
        var_df['old_idx'] = var_df.index
        var_df.set_index(scrna_var_key,inplace=True)
        adata = anndata.AnnData(X=adata.X, obs=adata.obs, var=var_df,obsm=adata.obsm, varm=adata.varm)

    if genes_to_use != 'all':
        # subset to genes intersecting with selected genes, possibly HVGs or those in ST 
        adata  = adata[:,list(set(adata.var.index).intersection(set(genes_to_use)))]
    
    # adata = adata_scrna.raw.to_adata()
    
    grouped = adata.obs.groupby(cluster_key)
    
    expr_avg = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = adata.X[idx]
        expr_avg[group] = np.ravel(X.mean(axis=0, dtype=np.float64))

    # binarized cell by cell-type matrix
    ct_identity = pd.get_dummies(adata.obs[cluster_key]).astype(int)
    # take column average  (bug fix
    ct_identity = (ct_identity/ np.ravel(ct_identity.sum(axis=0)))
    return adata, expr_avg, ct_identity


def get_ref_for_spatialloc(adata,cluster_key):
    grouped = adata.obs.groupby(cluster_key)
    
    expr_avg = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = adata.X[idx]
        expr_avg[group] = np.ravel(X.mean(axis=0, dtype=np.float64))

    # binarized cell by cell-type matrix
    ct_identity = pd.get_dummies(adata.obs[cluster_key]).astype(int)
    # take column average  (bug fix
    ct_identity = (ct_identity/ np.ravel(ct_identity.sum(axis=0)))
    return expr_avg, ct_identity



def get_ct_props_in_ref(adata_scrna, B,  ct_key="Cell class (determined from clustering of all cells)"):
    # input AnnData object
    # input B which contains cell-types we want to get proportions for in columns 
    
    adata_scrna = adata_scrna[adata_scrna.obs[ct_key].isin(B.columns),].copy()

    
    return adata_scrna.obs[ct_key].value_counts() / adata_scrna.shape[0]



def get_intrasample_spatial_dist(adata_spatial, spatial_key='spatial',recompute=False,nn_only=True,n_neigh=10):
    # spatial_sample is  anndata objects with spaital coordinates in adata.obsm[spatial_key]
    # cost_sp = distance_matrix(adata_spatial.obsm[spatial_key], adata_spatial.obsm[spatial_key]) # euclidean distance
    
    # if n_neigh is not None:
    #     # truncate to neighbors
    #     sq.gr.spatial_neighbors(adata_spatial, coord_type="generic",n_neighs=10) 
    if adata_spatial.shape[0] < (n_neigh+1):
        n_neigh = adata_spatial.shape[0] - 1

    spa_NNconn = np.ones((adata_spatial.shape[0],adata_spatial.shape[0]))
    if nn_only: # can use all or only nearest neighbors
        if (recompute or ('spatial_connectivities'  not in adata_spatial.obsp.keys()) or (adata_spatial.uns['spatial_neighbors']['params']['n_neighbors']!=n_neigh)):
            sq.gr.spatial_neighbors(adata_spatial, spatial_key=spatial_key,coord_type="generic",n_neighs=n_neigh) 
        
        # this is not symmetric, every column sums to 10 (since 10neighbors) but every row doesnt
        # spa_NNdist = adata_spatial.obsp['spatial_distances'].toarray() 
        spa_NNconn = adata_spatial.obsp['spatial_connectivities'].toarray() # 10NN connectivity 
    
    spadist = squareform(pdist(adata_spatial.obsm[spatial_key])) # distance (spatial) between every pari of cells

    # october 2024 debug update
    # print(np.allclose(spadist, spadist.T)) # check if symmetric
    # print(spa_NNconn.sum(axis=0)) # sums to k (for k-NN) 
    # print(spa_NNconn.sum(axis=1)) # does not sum to k

    # differences from cespgrn
    # 1. didnt make k-NN symmetric (that seems odd )
    # 2. didn't calculate and use shortest path distances but directly used only the k-NN 
    return spadist, spa_NNconn
 
# same slice wt
def get_gauss_kernel_wt(adata_spa,spatial_key='spatial',
                                nn_only=True,min_wt=0.0001, bandwidth=0.01,n_spatial_neigh=10,
                                recompute=False, weight_spatial=1.0):
    if adata_spa.shape[0] < (n_spatial_neigh+1):
        n_spatial_neigh = adata_spa.shape[0] - 1

    spa_dist, spa_NNconn = get_intrasample_spatial_dist(adata_spa, spatial_key=spatial_key,n_neigh=n_spatial_neigh,recompute=recompute)
    spa_dist = spa_dist/np.max(spa_dist)  # scale to [0,1]

    mdis = 0.5 * bandwidth * np.median(spa_dist)

    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)

    if nn_only:
        kernel_wt = np.multiply(kernel_wt,spa_NNconn) # restrict to NN only 
    
    kernel_wt[kernel_wt<min_wt]=0.0

    np.fill_diagonal(kernel_wt, 1.0) # uncomment for previous verions - Changed and then Rechanged on Apr 10, 2025 
    # print(kernel_wt.sum(axis=0))
    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0, keepdims=True)) # renormalize to 1 
    # print(kernel_wt.sum(axis=0))
    # np.fill_diagonal(kernel_wt, weight_spatial) 
    # print(kernel_wt.sum(axis=0))
    return weight_spatial * kernel_wt # , spa_NNconn


def get_gauss_kernel_3D(spa_dist, nneigh3D, bandwidth,  min_wt = 0.0001, nn_only=True):
    spa_dist =  (spa_dist/np.max(spa_dist))
    # print(spa_dist)
    mdis = 0.5 * bandwidth * np.median(spa_dist)
    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)
    # print(spa_dist)
    n = spa_dist.shape[1]
    if nn_only:
        spa_dist_mask = spa_dist.copy()
        np.fill_diagonal(spa_dist_mask, np.inf)
        sorted_indices = np.argsort(spa_dist_mask, axis=0)
        mask = np.zeros_like(spa_dist_mask, dtype=np.float32)
        cols = np.arange(n)
        mask[sorted_indices[:nneigh3D, cols], cols] = 1.0
        kernel_wt = kernel_wt * mask
    kernel_wt[kernel_wt<min_wt]=0.0

    np.fill_diagonal(kernel_wt, 1.0) # uncomment for previous verions - Changed and then Rechanged on Apr 10, 2025 
    # print(kernel_wt.sum(axis=0))
    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0, keepdims=True)) # renormalize to 1 
    # print(kernel_wt.sum(axis=0))
    # np.fill_diagonal(kernel_wt, weight_spatial) 
    # print(kernel_wt.sum(axis=0))
    return  kernel_wt # , spa_NNconn

    
def get_gauss_kernel_wt_neighslice(spa_dist,# adata_spa,spatial_key='spatial',
                                nn_only=True,min_wt=0.0001, bandwidth=0.01,n_spatial_neigh=10,
                                recompute=False, weight_spatial=1.0):
    # if adata_spa.shape[0] < (n_spatial_neigh+1):
    #     n_spatial_neigh = adata_spa.shape[0] - 1

    # spa_dist, spa_NNconn = get_intrasample_spatial_dist(adata_spa, spatial_key=spatial_key,n_neigh=n_spatial_neigh,recompute=recompute)
    # spa_dist = spa_dist/np.max(spa_dist)  # scale to [0,1]
    spa_dist = spa_dist/np.sum(spa_dist)
    n = spa_dist.shape[1]
    mdis = 0.5 * bandwidth * np.median(spa_dist)

    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)

    if nn_only:
        # select top k - neighbors 
        # kernel_wt = np.multiply(kernel_wt,spa_NNconn) # restrict to NN only 
        spa_dist_mask = spa_dist.copy()
        np.fill_diagonal(spa_dist_mask, np.inf)
        sorted_indices = np.argsort(spa_dist_mask, axis=0)
        mask = np.zeros_like(spa_dist_mask, dtype=np.float32)
        cols = np.arange(n)
        mask[sorted_indices[:n_spatial_neigh, cols], cols] = 1.0
        kernel_wt = kernel_wt * mask


    kernel_wt[kernel_wt<min_wt]=0.0

    np.fill_diagonal(kernel_wt, 1.0) # uncomment for previous verions - Changed and then Rechanged on Apr 10, 2025 
    # print(kernel_wt.sum(axis=0))
    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0, keepdims=True)) # renormalize to 1 
    # print(kernel_wt.sum(axis=0))
    # np.fill_diagonal(kernel_wt, weight_spatial) 
    # print((kernel_wt>0).sum(axis=0))
    return weight_spatial * kernel_wt # , spa_NNconn



def gaussian_kernel_for3d(x, sigma):
    """
    Returns Gaussian kernel values for a given point x,
    with standard deviation sigma and mean 0.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)



def permutation_test_local_geary(kernel_wt, Dsq, X1, X2, nsim=500, pval_cutoff=0.05, seed=0,batch_sz=10):
    # is_same_slice indicates whether we're computing geary C for same or adjacent slices 
    # means we need to set diagonal to 0 during perturbation (as we dont want to perturb the current cell)
    is_same_slice = torch.equal(X1, X2)
    
    nspots_src, ngenes      = X1.shape
    nspots_tgt, ngenes      = X2.shape
    # X =torch.from_numpy(X).to(device) - alredy tensro

    # kernel_wt ->  (n_obs, n_obs, bw, n_neigh)
    # Dsq -> n_obs, n_obs 
    n_bw      = kernel_wt.size(2)
    n_neigh   = kernel_wt.size(3) 

    # geary - C 
    obs_C    = (kernel_wt * Dsq.unsqueeze(-1).unsqueeze(-1)).sum(dim=0) # ( n_obs, bw, n_neigh)
    counts = torch.zeros_like(obs_C) # accumulate counts for batch processing

    # permute 
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for start in range(0, nsim, batch_sz):
        end = min(start + batch_sz, nsim)
        batch_n = end - start

        rand_vals = torch.rand((batch_n, nspots_src), generator=gen, device=device)
        perm_ids     = rand_vals.argsort(dim=1)           # (nsim, nspots)

        Dsq_perm = Dsq[perm_ids]

        # swap the 0 entries with the diagonal entires if same slice
        if is_same_slice:
            mask_zero = Dsq_perm == 0.0 # (nsim, n, n)
            #  assume exactly one zero per column per sim
            i_zero = mask_zero.float().argmax(dim=1) #  (nsim, n)
            sim_idx = torch.arange(batch_n, device=Dsq_perm.device).unsqueeze(1)  # (nsim,1)
            col_idx = torch.arange(nspots_tgt,    device=Dsq_perm.device).unsqueeze(0)  # (1,n)
            diag_vals    = torch.diagonal(Dsq_perm, dim1=1, dim2=2) # (nsim, n_tgt)
            Dsq_perm[sim_idx, i_zero, col_idx] = diag_vals
            Dsq_perm[sim_idx, col_idx, col_idx] = 0.0

        # C_perm = (kernel_wt.unsqueeze(0) * # (1, n_src, n_tgt, n_bw,  n_neigh)
        #           Dsq_perm.unsqueeze(-1).unsqueeze(-1) #  (batch_n, n_src, n_tgt,  1, 1)
        #          ).sum(dim=1)   #  (batch_n,  n_tgt, n_bw, n_neigh)
        
        # double check if this line is correct!! 
        # C_perm = torch.tensordot(Dsq_perm, kernel_wt, dims=([1], [0]))
        Dsq_perm = Dsq_perm.to(kernel_wt.dtype)
        C_perm = torch.einsum('bij,ijkm->bjkm', Dsq_perm, kernel_wt)

        counts += (C_perm <= obs_C.unsqueeze(0)).float().sum(dim=0) #   (batch_n,  n_tgt, n_bw, n_neigh) ( 1, n_tgt, bw, n_neigh)

    pvals = counts / nsim
    
    obs_C[pvals > pval_cutoff] = 1e20

    # print(obs_C_bin)
    return obs_C, pvals
    

def permutation_test_local_geary_fixedbw(kernel_wt, Dsq, X1, X2, nsim=500, pval_cutoff=0.05, seed=0, batch_sz=2):
    # is_same_slice indicates whether we're computing geary C for same or adjacent slices 
    # means we need to set diagonal to 0 during perturbation (as we dont want to perturb the current cell)
    is_same_slice = torch.equal(X1, X2)
    
    nspots_src, ngenes      = X1.shape
    nspots_tgt, ngenes      = X2.shape
    # X =torch.from_numpy(X).to(device) - alredy tensro

    # kernel_wt ->  (n_obs, n_obs,  n_neigh)
    # Dsq -> n_obs, n_obs 
    # n_bw      = kernel_wt.size(2)
    n_neigh   = kernel_wt.size(2) 

    # geary - C 
    obs_C    = (kernel_wt * Dsq.unsqueeze(-1)).sum(dim=0) # (n_obs, n_obs,  n_neigh) ->  (n_tgt,  n_neigh)

    counts = torch.zeros_like(obs_C) # accumulate counts for batch processing

    # permute 
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # batch permutations 
    for start in range(0, nsim, batch_sz):
        end = min(start + batch_sz, nsim)
        batch_n = end - start

        rand_vals = torch.rand((batch_n, nspots_src), generator=gen, device=device)
        perm_ids     = rand_vals.argsort(dim=1) 

        Dsq_perm = Dsq[perm_ids] # shape (batch_n, n_src, n_tgt)

        # make sure current spot isnt swapped 
        if is_same_slice:
            mask_zero = Dsq_perm == 0.0 # (batch_n, n_src, n_src)
            i_zero = mask_zero.float().argmax(dim=1) #  (batch_n, n)
            sim_idx = torch.arange(batch_n, device=Dsq_perm.device).unsqueeze(1)
            col_idx = torch.arange(nspots_tgt,    device=Dsq_perm.device).unsqueeze(0)
            diag_vals    = torch.diagonal(Dsq_perm, dim1=1, dim2=2) # get non-zero diagonal 
            Dsq_perm[sim_idx, i_zero, col_idx] = diag_vals  # move to current 0 position
            Dsq_perm[sim_idx, col_idx, col_idx] = 0.0
        
        C_perm = (kernel_wt.unsqueeze(0) * # (1, n_src, n_tgt,  n_neigh)
                  Dsq_perm.unsqueeze(-1) #  (batch_n, n_src, n_tgt,  1)
                 ).sum(dim=1)   #  (batch_n,  n_tgt,  n_neigh)
        
        counts += (C_perm <= obs_C.unsqueeze(0)).float().sum(dim=0) #  (batch_n,  n_tgt,  n_neigh) <=  (1, n_tgt,  n_neigh) -> (n_tgt,  n_neigh) 

    pvals = counts / nsim # (n_tgt,  n_neigh) 

    obs_C[pvals > pval_cutoff] = 1e20

    # print(obs_C_bin)
    return obs_C, pvals


def get_varbw_spatial_nbd(Xarr1, Xarr2, spacoord1, spacoord2, var_nspaneigh,var_bw, min_wt, # is_self, 
                          nsim=1000, pval_cutoff=0.05,Dspa=None, method='moscot', gt_align=False):#, DSpaConn=None ):
    
    # means we need to set diagonal to 0 during perturbation (as we dont want to perturb the current cell)
    is_same_slice = torch.equal(Xarr1, Xarr2)
   

    n_tgt = Xarr2.shape[0] # slice we're interedted in 
    n_src = Xarr1.shape[0]

    neigh_arr = torch.tensor(np.minimum(var_nspaneigh, n_src-1),device=device)

    Dsq =  torch.cdist(Xarr1, Xarr2, p=2).pow(2).to(device)  # (adata.X.unsqueeze(1) - adata.X.unsqueeze(0)).pow(2).sum(dim=2) # common across bws, neighs so compute outside 
    Dsq.masked_fill_(Dsq < 1e-10, 0.0)
    
    if Dspa is None:
        Dspa =  torch.cdist(spacoord1, spacoord2, p=2).to(device) # (adata.obsm['spatial'].unsqueeze(1) - adata.obsm['spatial'].X.unsqueeze(0)).pow(2).sum(dim=2)
        Dspa.masked_fill_(Dspa < 1e-7, 0.0)
        Dspa /= Dspa.max()
    # Dspa = Dspa.clamp_(min=1e-10)
    #Dspa.masked_fill_(Dspa < 1e-10, 0.0)
    tmp_med = torch.median(Dspa)
    
    # else:
    #     Dspa_nan = Dspa.clone()
    #     Dspa_nan[~torch.isfinite(Dspa_nan)] = float('nan')
    #     # then take the nan‐aware median
    #     tmp_med = torch.nanmedian(Dspa_nan)

    bw_arr = torch.tensor(var_bw,device=device)

    if gt_align:
        kernel_all_bw = Dspa.detach().clone()
    else:
        if ((not is_same_slice) and (method == 'slat')):
            kernel_all_bw = torch.exp(- Dspa.unsqueeze(-1).pow(2)) # for slat dont divide
        else:
            # (n_bw,)
            mdis = 0.5 * bw_arr * tmp_med

            kernel_all_bw = torch.exp(- Dspa.unsqueeze(-1).pow(2)/ mdis.unsqueeze(0).unsqueeze(0))  # (n_obs, n_obs, bw )
            
    idx   = torch.argsort(Dspa, dim=0) # row indices that would sort the column
    ranks = torch.argsort(idx,   dim=0)  # Here  take the inverse of that permutation. If idx[p,j] = i, that means “row isits in positionp of the sorted list for column j
    if is_same_slice:
        mask_all_neigh = (ranks.unsqueeze(-1) < (neigh_arr+1)) # (n_obs, n_obs, n_neigh)
    else:
        mask_all_neigh = (ranks.unsqueeze(-1) < neigh_arr)
    # if DSpaConn is not None:
    #     mask_all_neigh = mask_all_neigh * DSpaConn.unsqueeze(-1) # (n_obs, n_obs, n_neigh)

    kernel_all = kernel_all_bw.unsqueeze(-1) * mask_all_neigh.unsqueeze(-2) # (n_obs, n_obs, bw, 1) # (n_obs, n_obs, 1, n_neigh)
    
    # → (n_obs, n_obs, n_bw, n_neigh)
    kernel_all = kernel_all.masked_fill(kernel_all < min_wt, 0.0)  # set min_wt condition
    
    if is_same_slice:
        kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True)) # (n_obs, n_obs, bw, n_neigh)
    else:
        kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True) + 0.000001) 

    obs_C_bin, pvals = permutation_test_local_geary(kernel_all, Dsq, Xarr1, Xarr2, nsim=nsim, pval_cutoff=pval_cutoff)

    sig_any = (pvals <= pval_cutoff).any(dim=(1,2)) # n_tgt 

    flat  = obs_C_bin.view(n_tgt, -1) #  (n_obs, bw x n_neigh)
    best_par  = torch.argmin(flat, dim=1) 
    best_bw_idx  = best_par // neigh_arr.size(0)
    best_nn_idx  = best_par %  neigh_arr.size(0)

    i_idx = torch.arange(n_src, device=device)[:,None].expand(n_src, n_tgt)
    j_idx = torch.arange(n_tgt, device=device)[None,:].expand(n_src, n_tgt)
    bw_idx = best_bw_idx[None,:].expand(n_src, n_tgt)
    nn_idx = best_nn_idx[None,:].expand(n_src, n_tgt)

    kernel_all = kernel_all[i_idx, j_idx, bw_idx, nn_idx] # .copy()  # → (n_src, n_obs)
    
    # extract correct pvals,  geary, neighs 
    idx = torch.arange(n_tgt, device=device)
    best_localC = obs_C_bin[idx, best_bw_idx, best_nn_idx]  
    best_pvals  = pvals[idx, best_bw_idx, best_nn_idx]
    best_bw = bw_arr[best_bw_idx]
    best_neigh = neigh_arr[best_nn_idx]

    kernel_all[:, ~sig_any] = 0.0 # if nothing is significant, set to 0
    best_neigh[~sig_any] = 0

    # check if same_slice and sum == 0, set diag to 1 
    if is_same_slice:
        col_sums  = kernel_all.sum(dim=0)          # (n_tgt,)
        zero_cols = (col_sums == 0)                   # BoolTensor (n_tgt,)

        if zero_cols.any():
            # index of all targets
            idx = torch.arange(n_tgt, device=device)
            # pick out the ones that are zero
            bad = idx[zero_cols]                      # e.g. tensor([2,5,17], ...)
            # set those diagonal entries to 1
            kernel_all[bad, bad] = 1.0

    
    # kernel_all = torch.nan_to_num(kernel_all, nan=0.0)
    # print(kernel_all.sum(dim=0))

    # if is_same_slice: # means its the same slice -> if its the same slice make sure to use the current spot, else use not spots from that slice
    #     kernel_all.fill_diagonal_(1.0)

    # # renormalize 
    # if is_same_slice:
    #     kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True)) # (n_obs, n_obs, bw, n_neigh)
    # else:
    #     kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True) + 0.000001) 
    
    return kernel_all.cpu().numpy(), best_localC.cpu().numpy(), best_bw.cpu().numpy(), best_neigh.cpu().numpy(), best_pvals.cpu().numpy()



def get_align_slat(slice_order, slice_list):
    # adata_concat = slice_list[0].concatenate(slice_list[1:])
    # for colname in adata_concat.obs.columns:
    #     if adata_concat.obs[colname].isnull().any():
    #         adata_concat.obs[colname] = adata_concat.obs[colname].fillna(0)
            
    Palign = {} 


    
    for currslice_idx in range(len(slice_order)):
        for other_slice in range(len(slice_order)):
        
            if other_slice != currslice_idx:
                adata2 = slice_list[currslice_idx].copy()
                adata1 = slice_list[other_slice].copy()
                Cal_Spatial_Net(adata1, k_cutoff=10, model='KNN')
                Cal_Spatial_Net(adata2, k_cutoff=10, model='KNN')  
                
                edges, features = load_anndatas([adata1, adata2], feature='DPCA',check_order=False)  
                embd0, embd1, time = run_SLAT(features, edges)
                best, index, distance = spatial_match([embd0, embd1], adatas=[adata1,adata2], reorder=False)


                A = np.full((adata1.shape[0], adata2.shape[0]), np.inf, dtype=distance.dtype)

                
                cols = np.repeat(np.arange(adata2.shape[0]), 20)   # shape (2000,)
                rows = index.flatten()                   # shape (2000,)

                A[rows, cols] = distance.flatten()  
                
                
                # NNmask = (np.isfinite(A)).astype(float)

                # # 5) replace the +inf entries with a large “far” distance
                # #    here we take per‐column max finite + 1e3 (just an example buffer)
                # max_finite = np.where(np.isfinite(A), A, 0.0).max(axis=0, keepdims=True)
                # A = np.where(np.isfinite(A), A, max_finite + 1e3)

                # 6) now feed A and NNmask into the *exact* same Gaussian‐kernel logic as get_gauss_kernel_wt
                #    I’ve just pulled out the core of that function so you can reuse it:

                # Dspa_masked = A.clone()
                # Dspa_masked[~torch.isfinite(Dspa_masked)] = -float('inf')
                # col_max, _ = Dspa_masked.max(dim=0)  # shape (n_obs,)
                # A = A/col_max
                
                Palign[(str(slice_order[other_slice]),str(slice_order[currslice_idx]))] = A # convert alignment prob into distance 
     
    return Palign



def get_align_moscot(slice_order, slice_list, min_wt=0.0001):
    adata_concat = slice_list[0].concatenate(slice_list[1:])
    for colname in adata_concat.obs.columns:
        if adata_concat.obs[colname].isnull().any():
            adata_concat.obs[colname] = adata_concat.obs[colname].fillna(0)
            
    Palign = {} 
    Palign_orig = {}
    for currslice_idx in range(len(slice_order)):   
        ap_curr = AlignmentProblem(adata=adata_concat)
        ap_curr = ap_curr.prepare(batch_key="batch", policy="star",reference=str(currslice_idx))
        ap_curr = ap_curr.solve()
        
        # extract the transition matrices from ap
        for other_slice in range(len(slice_order)):
            if other_slice != currslice_idx:
                # print(str(other_slice),str(currslice_idx))
                Palign_tmp = np.asarray(ap_curr.solutions[(str(other_slice),str(currslice_idx))].transport_matrix)
                # each row (i.e. sum across all columns) is scaled by number of rows i.e. 1/number of rows
                # similarly each column (i.e. sum across all rows) is scaled by number of columns i.e. 1/number of columns
                # we want each column (target slice) to sum to 1 
                # i.e. we want each columns o.e. sum across all rows to be 1, so we scale by multiplying by the number of columns 
                
                Palign_tmp = Palign_tmp * Palign_tmp.shape[1] # every columns sums to 1 
                # Palign_tmp = 1-Palign_tmp
                # Palign_tmp /= Palign_tmp.max()
                Palign_orig[(str(slice_order[other_slice]),str(slice_order[currslice_idx]))] = Palign_tmp.copy()
                P_norm = Palign_tmp / Palign_tmp.max()
                P_safe = np.clip(P_norm, 0.00001, 1.0)
                d      = -np.log(P_safe)
                d      = d / d.max()   # if you want to rescale into [0,1]

                Palign[(str(slice_order[other_slice]),str(slice_order[currslice_idx]))] = d # convert alignment prob into distance 
     
    return Palign , Palign_orig





def get_align_moscot_one( slice_list, min_wt=0.0001):
    adata_concat = slice_list[0].concatenate(slice_list[1:])
    for colname in adata_concat.obs.columns:
        if adata_concat.obs[colname].isnull().any():
            adata_concat.obs[colname] = adata_concat.obs[colname].fillna(0)
            
    Palign = {} 
    Palign_orig = {}
    
    currslice_idx = len(slice_list)-1

    ap_curr = AlignmentProblem(adata=adata_concat)
    ap_curr = ap_curr.prepare(batch_key="batch", policy="star",reference=str(currslice_idx))
    ap_curr = ap_curr.solve()
    
    
    # extract the transition matrices from ap
    for other_slice in range(len(slice_list)):
        if other_slice != currslice_idx:
            # print(str(other_slice),str(currslice_idx))
            Palign_tmp = np.asarray(ap_curr.solutions[(str(other_slice),str(currslice_idx))].transport_matrix)
            # each row (i.e. sum across all columns) is scaled by number of rows i.e. 1/number of rows
            # similarly each column (i.e. sum across all rows) is scaled by number of columns i.e. 1/number of columns
            # we want each column (target slice) to sum to 1 
            # i.e. we want each columns o.e. sum across all rows to be 1, so we scale by multiplying by the number of columns 
            
            Palign_tmp = Palign_tmp * Palign_tmp.shape[1] # every columns sums to 1 
            # Palign_tmp = 1-Palign_tmp
            # Palign_tmp /= Palign_tmp.max()
            Palign_orig[(str(other_slice),str(currslice_idx))] = Palign_tmp.copy()
            P_norm = Palign_tmp / Palign_tmp.max()
            P_safe = np.clip(P_norm, 0.00001, 1.0)
            d      = -np.log(P_safe)
            d      = d / d.max()   # if you want to rescale into [0,1]

            Palign[(str(other_slice),str(currslice_idx))] = d # convert alignment prob into distance 
    
    return Palign , Palign_orig



def pairwise_row_pearson(X, Y):
    """
    Efficiently compute Pearson correlation between every pair of rows in X and Y.

    Parameters:
    - X: (n, d) array
    - Y: (m, d) array

    Returns:
    - corr: (n, m) array, Pearson correlations
    """
    # Center rows by subtracting the mean
    X_centered = X  - X.mean(axis=1, keepdims=True)
    Y_centered = Y  - Y.mean(axis=1, keepdims=True)

    # Normalize rows to unit length
    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_centered, axis=1, keepdims=True)

    # Avoid division by zero
    X_norm[X_norm == 0] = 1e-16
    Y_norm[Y_norm == 0] = 1e-16

    X_normalized = X_centered / X_norm
    Y_normalized = Y_centered / Y_norm

    # Compute correlations via dot product
    corr = X_normalized @ Y_normalized.T

    return corr


def pairwise_row_cosine_distance(X, Y):
    """
    Compute cosine distance between every pair of rows in X and Y efficiently using NumPy.

    Parameters:
    - X: (n, d) array
    - Y: (m, d) array

    Returns:
    - distances: (n, m) array, cosine distances
    """
    # Normalize rows to unit vectors
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)

    X_normalized = X / np.clip(X_norm, 1e-16, None)
    Y_normalized = Y / np.clip(Y_norm, 1e-16, None)

    # Compute cosine similarity
    cosine_similarity = X_normalized @ Y_normalized.T

    # Convert similarity to distance
    distances = cosine_similarity

    return distances


def get_spadist(X1,X2):
    diffs = X1[:, None, :] - X2[None, :, :]
    # sqrt of sum of squares along the last axis
    D = np.linalg.norm(diffs, axis=2)
    return D 



def get_align_modifiedfgw(slice_order, slice_list,adata_res_list, deconv_par=0.8, fgw_alpha=0.3,
            fgw_epsilon=1e-2,
            fgw_tau_a=0.99,
            fgw_tau_b = 0.99, min_wt=0.0001):
    # adata_res - deconv results 
    # artificailly set sim_slice = 0
    for tmp_idx in range(len(slice_order)):
        sc.pp.pca(slice_list[tmp_idx])
       

    adata_concat = slice_list[0].concatenate(slice_list[1:])
    for colname in adata_concat.obs.columns:
        if adata_concat.obs[colname].isnull().any():
            adata_concat.obs[colname] = adata_concat.obs[colname].fillna(0)
            
    Palign = {} 
    Palign_orig = {}
    for currslice in range(len(slice_order)): 
        currslice_idx = slice_order[currslice]
        fgw = FGWProblem(adata_concat)
        # PCA helped alot over raw feature space
        fgw = fgw.prepare(key="batch", policy="star",reference=str(currslice_idx),x_attr="X_pca",  y_attr="X_pca") # 
    
    
        for other_slice_idx in range(len(slice_order)):
            other_slice = slice_order[other_slice_idx]
            if other_slice != currslice_idx:
                obs_names_0 = fgw[str(other_slice), str(currslice_idx)].adata_src.obs_names
                obs_names_1 = fgw[str(other_slice), str(currslice_idx)].adata_tgt.obs_names

                # sq_eu = np.sum((fgw[other_slice, currslice_idx].adata_src.X[:, np.newaxis, :] - fgw[other_slice, currslice_idx].adata_tgt.X[np.newaxis, :, :]) ** 2, axis=2)
                # pearson_expr = 1-pairwise_row_cosine_distance(fgw[other_slice, currslice_idx].adata_src.X ,fgw[other_slice, currslice_idx].adata_tgt.X)
                
                pearson_expr = 1-pairwise_row_cosine_distance(fgw[str(other_slice), str(currslice_idx)].adata_src.obsm['X_pca'] ,fgw[str(other_slice), str(currslice_idx)].adata_tgt.obsm['X_pca'])
                print(pearson_expr)
                pearson_deconv = 1-pairwise_row_pearson(adata_res_list[other_slice_idx].values.T,adata_res_list[currslice].values.T)
                print(pearson_deconv)
                cost_linear_01 = (1-deconv_par)*pearson_expr + (deconv_par*pearson_deconv) # np.abs(rng.normal(size=(len(obs_names_0), len(obs_names_1))))
                # cost_quad_0 = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_src.X,fgw[other_slice, currslice_idx].adata_src.X)
                # cost_quad_1 = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_tgt.X,fgw[other_slice, currslice_idx].adata_tgt.X)
                
                # pearson_expr = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_src.X,fgw[other_slice, currslice_idx].adata_src.X)
                # pearson_deconv = 1-pairwise_row_pearson(adata_res[other_slice][0].values.T,adata_res[other_slice][0].values.T)
                # cost_quad_0 = (1-deconv_par)*pearson_expr + (deconv_par*pearson_deconv) # np.abs(rng.normal(size=(len(obs_names_0), len(obs_names_1))))
                
                # pearson_expr = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_tgt.X,fgw[other_slice, currslice_idx].adata_tgt.X)
                # pearson_deconv = 1-pairwise_row_pearson(adata_res[currslice_idx][0].values.T,adata_res[currslice_idx][0].values.T)
                # cost_quad_1 = (1-deconv_par)*pearson_expr + (deconv_par*pearson_deconv) # np.abs(rng.normal(size=(len(obs_names_0), len(obs_names_1))))
                
                cost_quad_0 = get_spadist(fgw[str(other_slice), str(currslice_idx)].adata_src.obsm['spatial'],fgw[str(other_slice), str(currslice_idx)].adata_src.obsm['spatial'])
                cost_quad_1 = get_spadist(fgw[str(other_slice), str(currslice_idx)].adata_tgt.obsm['spatial'],fgw[str(other_slice), str(currslice_idx)].adata_tgt.obsm['spatial'])
                
                # print(cost_linear_01, cost_quad_0, cost_quad_1)

                cm_linear = pd.DataFrame(data=cost_linear_01, index=obs_names_0, columns=obs_names_1)
                cm_quad_0 = pd.DataFrame(data=cost_quad_0, index=obs_names_0, columns=obs_names_0)
                cm_quad_1 = pd.DataFrame(data=cost_quad_1, index=obs_names_1, columns=obs_names_1)
                

                fgw[str(other_slice), str(currslice_idx)].set_xy(cm_linear, tag="cost_matrix")
                fgw[str(other_slice), str(currslice_idx)].set_x(cm_quad_0, tag="cost_matrix")
                fgw[str(other_slice), str(currslice_idx)].set_y(cm_quad_1, tag="cost_matrix")
                
            
        fgw = fgw.solve(alpha=fgw_alpha,
            epsilon=fgw_epsilon,
            tau_a=fgw_tau_a,
            tau_b = fgw_tau_b)
            # rank = -1) #0.99)
            # rank = 10)
        # extract the transition matrices from ap
        for other_slice in range(len(slice_order)):
            if other_slice != currslice_idx:
                # print(str(other_slice),str(currslice_idx))
                Palign_tmp = np.asarray(fgw.solutions[(str(other_slice),str(currslice_idx))].transport_matrix)
                print(Palign_tmp)

                Palign_tmp = Palign_tmp * Palign_tmp.shape[1]

                print(Palign_tmp)

                Palign_orig[(str(slice_order[other_slice]),str(slice_order[currslice_idx]))] = Palign_tmp.copy()

                P_norm = Palign_tmp / Palign_tmp.max()
                P_safe = np.clip(P_norm, 0.00001, 1.0)
                d      = -np.log(P_safe)
                d      = d / d.max()   # if you want to rescale into [0,1]


                # Palign_tmp = 1-Palign_tmp
                # Palign_tmp /= Palign_tmp.max()
                Palign[(str(slice_order[other_slice]),str(slice_order[currslice_idx]))] = d # convert alignment prob into distance 
     
    return Palign , Palign_orig









def get_align_modifiedfgw_one( slice_list,adata_res, deconv_par=0.3, fgw_alpha=0.3,
            fgw_epsilon=1e-2,
            fgw_tau_a=0.99,
            fgw_tau_b = 0.99, min_wt=0.0001):
    # adata_res - deconv results 
    # artificailly set sim_slice = 0
    for tmp_idx in range(len(slice_list)):
        sc.pp.pca(slice_list[tmp_idx])
       

    adata_concat = slice_list[0].concatenate(slice_list[1:])
    for colname in adata_concat.obs.columns:
        if adata_concat.obs[colname].isnull().any():
            adata_concat.obs[colname] = adata_concat.obs[colname].fillna(0)
            
    Palign = {} 
    Palign_orig = {}
    
    currslice_idx = len(slice_list) - 1 
    fgw = FGWProblem(adata_concat)
    # PCA helped alot over raw feature space
    fgw = fgw.prepare(key="batch", policy="star",reference=str(currslice_idx),x_attr="X_pca",  y_attr="X_pca") # 


    for other_slice_idx in range(len(slice_list)):
        other_slice = slice_order[other_slice_idx]
        if other_slice != currslice_idx:
            obs_names_0 = fgw[str(other_slice), str(currslice_idx)].adata_src.obs_names
            obs_names_1 = fgw[str(other_slice), str(currslice_idx)].adata_tgt.obs_names

            # sq_eu = np.sum((fgw[other_slice, currslice_idx].adata_src.X[:, np.newaxis, :] - fgw[other_slice, currslice_idx].adata_tgt.X[np.newaxis, :, :]) ** 2, axis=2)
            # pearson_expr = 1-pairwise_row_cosine_distance(fgw[other_slice, currslice_idx].adata_src.X ,fgw[other_slice, currslice_idx].adata_tgt.X)
            
            pearson_expr = 1-pairwise_row_cosine_distance(fgw[str(other_slice), str(currslice_idx)].adata_src.obsm['X_pca'] ,fgw[str(other_slice), str(currslice_idx)].adata_tgt.obsm['X_pca'])
            
            pearson_deconv = 1-pairwise_row_cosine_distance(adata_res_list[other_slice_idx].values.T,adata_res_list[currslice].values.T)
            cost_linear_01 = (1-deconv_par)*pearson_expr + (deconv_par*pearson_deconv) # np.abs(rng.normal(size=(len(obs_names_0), len(obs_names_1))))
            # cost_quad_0 = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_src.X,fgw[other_slice, currslice_idx].adata_src.X)
            # cost_quad_1 = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_tgt.X,fgw[other_slice, currslice_idx].adata_tgt.X)
            
            # pearson_expr = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_src.X,fgw[other_slice, currslice_idx].adata_src.X)
            # pearson_deconv = 1-pairwise_row_pearson(adata_res[other_slice][0].values.T,adata_res[other_slice][0].values.T)
            # cost_quad_0 = (1-deconv_par)*pearson_expr + (deconv_par*pearson_deconv) # np.abs(rng.normal(size=(len(obs_names_0), len(obs_names_1))))
            
            # pearson_expr = 1- pairwise_row_pearson(fgw[other_slice, currslice_idx].adata_tgt.X,fgw[other_slice, currslice_idx].adata_tgt.X)
            # pearson_deconv = 1-pairwise_row_pearson(adata_res[currslice_idx][0].values.T,adata_res[currslice_idx][0].values.T)
            # cost_quad_1 = (1-deconv_par)*pearson_expr + (deconv_par*pearson_deconv) # np.abs(rng.normal(size=(len(obs_names_0), len(obs_names_1))))
            
            cost_quad_0 = get_spadist(fgw[str(other_slice), str(currslice_idx)].adata_src.obsm['spatial'],fgw[str(other_slice), str(currslice_idx)].adata_src.obsm['spatial'])
            cost_quad_1 = get_spadist(fgw[str(other_slice), str(currslice_idx)].adata_tgt.obsm['spatial'],fgw[str(other_slice), str(currslice_idx)].adata_tgt.obsm['spatial'])
            
            # print(cost_linear_01, cost_quad_0, cost_quad_1)

            cm_linear = pd.DataFrame(data=cost_linear_01, index=obs_names_0, columns=obs_names_1)
            cm_quad_0 = pd.DataFrame(data=cost_quad_0, index=obs_names_0, columns=obs_names_0)
            cm_quad_1 = pd.DataFrame(data=cost_quad_1, index=obs_names_1, columns=obs_names_1)
            

            fgw[str(other_slice), str(currslice_idx)].set_xy(cm_linear, tag="cost_matrix")
            fgw[str(other_slice), str(currslice_idx)].set_x(cm_quad_0, tag="cost_matrix")
            fgw[str(other_slice), str(currslice_idx)].set_y(cm_quad_1, tag="cost_matrix")
            
        
    fgw = fgw.solve(alpha=fgw_alpha,
        epsilon=fgw_epsilon,
        tau_a=fgw_tau_a,
        tau_b = fgw_tau_b)
        # rank = -1) #0.99)
        # rank = 10)
    # extract the transition matrices from ap
    for other_slice in slice_order:
        if other_slice != currslice_idx:
            # print(str(other_slice),str(currslice_idx))
            Palign_tmp = np.asarray(fgw.solutions[(str(other_slice),str(currslice_idx))].transport_matrix)


            Palign_tmp = Palign_tmp * Palign_tmp.shape[1]

            Palign_orig[(str(other_slice),str(currslice_idx))] = Palign_tmp.copy()

            P_norm = Palign_tmp / Palign_tmp.max()
            P_safe = np.clip(P_norm, 0.00001, 1.0)
            d      = -np.log(P_safe)
            d      = d / d.max()   # if you want to rescale into [0,1]


            # Palign_tmp = 1-Palign_tmp
            # Palign_tmp /= Palign_tmp.max()
            Palign[(str(other_slice),str(currslice_idx))] = d # convert alignment prob into distance 
    
    return Palign , Palign_orig





def get_variable_bw(slice_order, slice_list, var_nspaneigh,var_bw, mode, 
                    Palign_dis=None,  min_wt=0.0001,method='moscot', gt_align=False):
    kernel_all = {}
    geary_metric = {}

    for currslice in range(len(slice_order)):
        currslice_idx = slice_order[currslice]

        adata = slice_list[currslice]
        Xarr = torch.tensor(adata.X).to(device)
        spacoord = torch.tensor(adata.obsm['spatial'])
        kernel_all[str(currslice_idx) + '_' + str(currslice_idx)], best_localC, bw_arr, neigh_arr, best_pvals =  get_varbw_spatial_nbd(Xarr, Xarr, spacoord, spacoord, var_nspaneigh,var_bw, min_wt)#, is_self  )

        geary_metric[str(currslice_idx) + '_' + str(currslice_idx)] = pd.DataFrame({
            'gearyc': best_localC,
            'bw':      bw_arr,
            'neigh':   neigh_arr,
            'is_sig':  best_pvals
        },index=list(range(adata.shape[0])))

        if mode == 'multislice':
            for other_slice_idx in range(len(slice_order)): # for 3D kernel we need to compute the geary metric across slices
                other_slice = slice_order[other_slice_idx]
                if other_slice != currslice_idx:
                    # print(other_slice, currslice_idx)
                    adata_other = slice_list[other_slice_idx]
                    Xarr_other = torch.tensor(adata_other.X).to(device)
                    spacoord_other = torch.tensor(adata_other.obsm['spatial'])
                    # get the 3D distance 
                    kernel_all[ str(other_slice) + '_' + str(currslice_idx)  ], best_localC, bw_arr, neigh_arr, best_pvals =  get_varbw_spatial_nbd(Xarr_other, Xarr, spacoord_other, spacoord, var_nspaneigh,var_bw, min_wt,  Dspa =  torch.from_numpy(Palign_dis[(str(other_slice),str(currslice_idx))]).to(device),method=method, gt_align=gt_align) #, DSpaConn=DSpaConn)

                    geary_metric[str(other_slice) + '_' + str(currslice_idx) ] = pd.DataFrame({
                        'gearyc': best_localC,
                        'bw':      bw_arr,
                        'neigh':   neigh_arr,
                        'is_sig':  best_pvals
                    },index=list(range(adata.shape[0])))
    return kernel_all, geary_metric




def get_fixed_bw(slice_order, slice_list, mode,  fixed_bw=0.01, fixed_nn=10, 
                 Palign_dis=None, min_wt=0.0001, method='moscot', gt_align=False):
    kernel_all = {}
    # geary_metric = {}

    for currslice in range(len(slice_order)):
        currslice_idx = slice_order[currslice]
        spacoord = torch.tensor(slice_list[currslice].obsm['spatial'])        
        Dspa =  torch.cdist(spacoord, spacoord, p=2).to(device) # (adata.obsm['spatial'].unsqueeze(1) - adata.obsm['spatial'].X.unsqueeze(0)).pow(2).sum(dim=2)
        Dspa.masked_fill_(Dspa < 1e-7, 0.0)
        Dspa /= Dspa.max()
        tmp_med = torch.median(Dspa)
        mdis = 0.5 * fixed_bw * tmp_med
        kernel_fixed_bw = torch.exp(- Dspa.pow(2)/ mdis)  # (n_obs, n_obs, bw )
        idx   = torch.argsort(Dspa, dim=0)             # (n_obs, n_obs)
        ranks = torch.argsort(idx,   dim=0)            # inverse permutation
        mask = (ranks < (fixed_nn + 1))                       # (n_obs, n_obs), bool
        kernel_fixed_bw = kernel_fixed_bw * mask # (n_obs, n_obs, n_bw)
        kernel_fixed_bw = kernel_fixed_bw.masked_fill(kernel_fixed_bw < min_wt, 0.0)
        kernel_all[ str(currslice_idx) + '_' + str(currslice_idx)  ] = (kernel_fixed_bw / kernel_fixed_bw.sum(dim=0, keepdim=True)).cpu().numpy()
    
        if mode == 'multislice':
            for other_slice in slice_order: # for 3D kernel we need to compute the geary metric across slices
                if other_slice != currslice_idx:
                    Dspa = torch.from_numpy(Palign_dis[(str(other_slice),str(currslice_idx))]).to(device)
                    
                    # # required for SLAT where distances not computed were set to inf
                    # Dspa_nan = Dspa.clone()
                    # Dspa_nan[~torch.isfinite(Dspa_nan)] = float('nan')
                    # # then take the nan‐aware median
                    # tmp_med = torch.nanmedian(Dspa_nan)

                    # tmp_med = torch.median(Dspa)
                    if gt_align:
                        kernel_fixed_bw = Dspa.detach().clone()
                    else:
                        if method == 'slat':
                            kernel_fixed_bw = torch.exp(- Dspa.pow(2))
                        else:
                            mdis = 0.5 * fixed_bw * tmp_med
                            kernel_fixed_bw = torch.exp(- Dspa.pow(2)/ mdis)
                            

                    
                      # (n_obs, n_obs, bw )
                    idx   = torch.argsort(Dspa, dim=0)             # (n_obs, n_obs)
                    ranks = torch.argsort(idx,   dim=0)            # inverse permutation
                    mask = (ranks < fixed_nn)                       # (n_obs, n_obs), bool
                    kernel_fixed_bw = kernel_fixed_bw * mask # (n_obs, n_obs, n_bw)
                    kernel_fixed_bw = kernel_fixed_bw.masked_fill(kernel_fixed_bw < min_wt, 0.0)
                    kernel_all[ str(other_slice) + '_' + str(currslice_idx)  ] = (kernel_fixed_bw / (kernel_fixed_bw.sum(dim=0, keepdim=True) + 0.000001)).cpu().numpy()           
    return kernel_all




def get_vartranscrbw_spatial_nbd(Xarr1, Xarr2, spacoord1, spacoord2, var_nspaneigh,fixed_bw, min_wt, # is_self, 
                          nsim=1000, pval_cutoff=0.05,Dspa=None,method='moscot',gt_align=False, batch_sz=2):#, DSpaConn=None ):
    
    # means we need to set diagonal to 0 during perturbation (as we dont want to perturb the current cell)
    is_same_slice = torch.equal(Xarr1, Xarr2)
   
    n_tgt = Xarr2.shape[0] # slice we're interested in 
    n_src = Xarr1.shape[0]

    neigh_arr = torch.tensor(np.minimum(var_nspaneigh, n_src-1),device=device)

    Dsq =  torch.cdist(Xarr1, Xarr2, p=2).pow(2).to(device)  # (adata.X.unsqueeze(1) - adata.X.unsqueeze(0)).pow(2).sum(dim=2) # common across bws, neighs so compute outside 
    Dsq.masked_fill_(Dsq < 1e-10, 0.0)
    if Dspa is None:
        Dspa =  torch.cdist(spacoord1, spacoord2, p=2).to(device) # (adata.obsm['spatial'].unsqueeze(1) - adata.obsm['spatial'].X.unsqueeze(0)).pow(2).sum(dim=2)
        Dspa.masked_fill_(Dspa < 1e-7, 0.0)
        Dspa /= Dspa.max()
    # Dspa = Dspa.clamp_(min=1e-10)
    
    tmp_med = torch.median(Dspa)
    
    # bw_arr = torch.tensor(var_bw,device=device)  # (n_bw,)
    
    if gt_align: # align corresponding spots dont use distance 
        kernel_all_bw = Dspa.detach().clone()
    else:
        if ((not is_same_slice) and (method == 'slat')):
            kernel_all_bw = torch.exp(- Dspa.pow(2)) # for slat dont divide
        else:
            # (n_bw,)
            mdis = 0.5 * fixed_bw * tmp_med
            kernel_all_bw = torch.exp(- Dspa.pow(2)/ mdis)  # (n_obs, n_obs)
            
    #kernel_all_bw = torch.exp(- Dspa.pow(2)/ mdis)  # (n_obs, n_obs)

    idx   = torch.argsort(Dsq, dim=0) # row indices that would sort the column
    ranks = torch.argsort(idx,   dim=0)  # Here  take the inverse of that permutation. If idx[p,j] = i, that means “row isits in positionp of the sorted list for column j
    if is_same_slice:
        mask_all_neigh = (ranks.unsqueeze(-1) < (neigh_arr+1)) # (n_obs, n_obs, n_neigh)
    else:
        mask_all_neigh = (ranks.unsqueeze(-1) < neigh_arr)
    # if DSpaConn is not None:
    #     mask_all_neigh = mask_all_neigh * DSpaConn.unsqueeze(-1) # (n_obs, n_obs, n_neigh)

    kernel_all = kernel_all_bw.unsqueeze(-1) * mask_all_neigh # (n_obs, n_obs,  1) ->  (n_obs, n_obs,  n_neigh)
    
    # → (n_obs, n_obs, n_bw, n_neigh)
    kernel_all = kernel_all.masked_fill(kernel_all < min_wt, 0.0)  # set min_wt condition
   
    # kernel_all = kernel_all / kernel_all.sum(dim=0, keepdim=True) # (n_obs, n_obs, bw, n_neigh)
    if is_same_slice:
        kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True)) # (n_obs, n_obs, bw, n_neigh)
    else:
        kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True) + 0.000001) 

    # (nspots_tgt,  neigh)
    obs_C_bin, pvals = permutation_test_local_geary_fixedbw(kernel_all, Dsq, Xarr1, Xarr2, nsim=nsim, pval_cutoff=pval_cutoff, batch_sz=batch_sz)

    ###########
    # try select largest number of transcr neighbors that is significant 
    cols = torch.arange(obs_C_bin.shape[1], device=device) 
    col_idx = cols.unsqueeze(0).expand(n_tgt, obs_C_bin.shape[1])
    mask_pval = pvals < pval_cutoff
    masked_idx = col_idx.masked_fill(~mask_pval, -1) 
    best_nn_idx = masked_idx.max(dim=1).values
    ################## 
    
    sig_any = (pvals <= pval_cutoff).any(dim=1) # n_tgt 

    # tmp_gearyc = obs_C_bin.clone() # uncommnet 
    # tmp_gearyc[tmp_gearyc == 0.0] = float('inf') # uncommnet 

    # best_nn_idx  = torch.argmin(tmp_gearyc, dim=1) # ntgt  # uncommnet  obs_C_bin
    
    i_idx = torch.arange(n_src, device=device)[:,None].expand(n_src, n_tgt)
    j_idx = torch.arange(n_tgt, device=device)[None,:].expand(n_src, n_tgt)
    nn_idx = best_nn_idx[None,:].expand(n_src, n_tgt)

    

    kernel_all = kernel_all[i_idx, j_idx,  nn_idx] # .copy()  # → (n_src, n_obs)

    # extract correct neighbors and p-value 
    
    idx = torch.arange(n_tgt, device=device)
    best_localC = obs_C_bin[idx,  best_nn_idx]  
    best_pvals  = pvals[idx, best_nn_idx]
    best_neigh = neigh_arr[best_nn_idx] 

    kernel_all[:, ~sig_any] = 0.0 # if nothing is significant, set to 0
    best_neigh[~sig_any] = 0

    # check if same_slice and sum == 0, set diag to 1 
    if is_same_slice:
        col_sums  = kernel_all.sum(dim=0)          # (n_tgt,)
        zero_cols = (col_sums == 0)                   # BoolTensor (n_tgt,)

        if zero_cols.any():
            # index of all targets
            idx = torch.arange(n_tgt, device=device)
            # pick out the ones that are zero
            bad = idx[zero_cols]                      # e.g. tensor([2,5,17], ...)
            # set those diagonal entries to 1
            kernel_all[bad, bad] = 1.0

    # import sys
    # with open('tmp_kernel.pickle','wb') as handle:
    #     pickle.dump(kernel_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('tmp_geary.pickle','wb') as handle:
    #     pickle.dump(obs_C_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('tmp_pvals','wb') as handle:
    #     pickle.dump(pvals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('tmp_Dsq','wb') as handle:
    #     pickle.dump(Dsq, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('tmp_Dspa','wb') as handle:
    #     pickle.dump(Dspa, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # sys.exit()

    
    
    #kernel_all = torch.nan_to_num(kernel_all, nan=0.0)
    # check if the columns sum to 1 
    # print(kernel_all.sum(dim=0))

    # if is_same_slice: # means its the same slice -> if its the same slice make sure to use the current spot, else use not spots from that slice
    #     kernel_all.fill_diagonal_(1.0)

    # if is_same_slice:
    #     kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True)) # (n_obs, n_obs, bw, n_neigh)
    # else:
    #     kernel_all = kernel_all / (kernel_all.sum(dim=0, keepdim=True) + 0.000001) 
   
    # renormalize 
    # kernel_all = kernel_all / kernel_all.sum(dim=0, keepdim=True)
    
    return kernel_all.cpu().numpy(), best_localC.cpu().numpy(),  best_neigh.cpu().numpy(), best_pvals.cpu().numpy()



def get_variabletranscr_bw(slice_order, slice_list,var_nspaneigh_transcr, mode, 
                           fixed_bw=0.01,Palign_dis=None,  min_wt=0.0001, method='moscot',
                           gt_align=False,batch_sz=2):
    kernel_all = {}
    geary_metric = {}

    for currslice in range(len(slice_order)):
        currslice_idx = slice_order[currslice]

        adata = slice_list[currslice]
        Xarr = torch.tensor(adata.X).to(device)
        spacoord = torch.tensor(adata.obsm['spatial'])
        kernel_all[str(currslice_idx) + '_' + str(currslice_idx)], best_localC,  neigh_arr, best_pvals =  get_vartranscrbw_spatial_nbd(Xarr, Xarr, spacoord, spacoord, var_nspaneigh_transcr,fixed_bw, min_wt, batch_sz=batch_sz)#, is_self  )

        geary_metric[str(currslice_idx) + '_' + str(currslice_idx)] = pd.DataFrame({
            'gearyc': best_localC,
            #'bw':      bw_arr,
            'neigh':   neigh_arr,
            'is_sig':  best_pvals
        },index=list(range(adata.shape[0])))

        if mode == 'multislice':
            for other_slice_idx in range(len(slice_order)): # for 3D kernel we need to compute the geary metric across slices
                other_slice = slice_order[other_slice_idx]
                if other_slice != currslice_idx:
                    print(other_slice, currslice_idx)
                    adata_other = slice_list[other_slice_idx]
                    Xarr_other = torch.tensor(adata_other.X).to(device)
                    spacoord_other = torch.tensor(adata_other.obsm['spatial'])
                    # get the 3D distance 
                    
                    kernel_all[ str(other_slice) + '_' + str(currslice_idx)  ], best_localC,  neigh_arr, best_pvals =  get_vartranscrbw_spatial_nbd(Xarr_other, Xarr, spacoord_other, spacoord, var_nspaneigh_transcr,fixed_bw, min_wt,  Dspa =  torch.from_numpy(Palign_dis[(str(other_slice),str(currslice_idx))]).to(device),method=method, gt_align=gt_align) #, DSpaConn=DSpaConn)

                    geary_metric[str(other_slice) + '_' + str(currslice_idx)] = pd.DataFrame({
                        'gearyc': best_localC,
                        #'bw':      bw_arr,
                        'neigh':   neigh_arr,
                        'is_sig':  best_pvals
                    },index=list(range(adata.shape[0])))
    return kernel_all, geary_metric


def get_fixed_transcrbw(slice_order, slice_list, mode,  fixed_bw=0.01, fixed_nn_self=10, fixed_nn_neigh=10, Palign_dis=None, min_wt=0.0001, method='moscot'):
    kernel_all = {}
    # geary_metric = {}

    for currslice in range(len(slice_order)):
        currslice_idx = slice_order[currslice]
        spacoord = torch.tensor(slice_list[currslice].obsm['spatial'])     
        
        Xarr = torch.tensor(slice_list[currslice].X).to(device)
        Dsq =  torch.cdist(Xarr, Xarr, p=2).pow(2).to(device)
        Dsq.masked_fill_(Dsq < 1e-10, 0.0)

        Dspa =  torch.cdist(spacoord, spacoord, p=2).to(device) # (adata.obsm['spatial'].unsqueeze(1) - adata.obsm['spatial'].X.unsqueeze(0)).pow(2).sum(dim=2)
        Dspa.masked_fill_(Dspa < 1e-7, 0.0)
        Dspa /= Dspa.max()
        tmp_med = torch.median(Dspa)

        mdis = 0.5 * fixed_bw * tmp_med
        kernel_fixed_bw = torch.exp(- Dspa.pow(2)/ mdis)  # (n_obs, n_obs, bw )
        
        idx   = torch.argsort(Dsq, dim=0)
        # idx   = torch.argsort(Dspa, dim=0)             # (n_obs, n_obs)
        ranks = torch.argsort(idx,   dim=0)            # inverse permutation
        
        mask = (ranks < (fixed_nn_self + 1))                       # (n_obs, n_obs), bool
        kernel_fixed_bw = kernel_fixed_bw * mask # (n_obs, n_obs, n_bw)
        kernel_fixed_bw = kernel_fixed_bw.masked_fill(kernel_fixed_bw < min_wt, 0.0)
        kernel_all[ str(currslice_idx) + '_' + str(currslice_idx)  ] = (kernel_fixed_bw / kernel_fixed_bw.sum(dim=0, keepdim=True)).cpu().numpy()
    
        if mode == 'multislice':
            for other_slice_idx in range(len(slice_order)): # for 3D kernel we need to compute the geary metric across slices
                other_slice = slice_order[other_slice_idx]
                if other_slice != currslice_idx:
                    Dspa = torch.from_numpy(Palign_dis[(str(other_slice),str(currslice_idx))]).to(device)
                    
                    # # required for SLAT where distances not computed were set to inf
                    # Dspa_nan = Dspa.clone()
                    # Dspa_nan[~torch.isfinite(Dspa_nan)] = float('nan')
                    # # then take the nan‐aware median
                    # tmp_med = torch.nanmedian(Dspa_nan)

                    # tmp_med = torch.median(Dspa)
                    if method == 'slat':
                        kernel_fixed_bw = torch.exp(- Dspa.pow(2))
                    else:
                        mdis = 0.5 * fixed_bw * tmp_med
                        kernel_fixed_bw = torch.exp(- Dspa.pow(2)/ mdis)
                        
                    Xarr_src = torch.tensor(slice_list[other_slice_idx].X).to(device)
                    Dsq =  torch.cdist(Xarr_src, Xarr, p=2).pow(2).to(device) 
                    Dsq.masked_fill_(Dsq < 1e-10, 0.0)
                    idx   = torch.argsort(Dsq, dim=0)  
                    # (n_obs, n_obs, bw )
                    # idx   = torch.argsort(Dspa, dim=0)             # (n_obs, n_obs)
                    ranks = torch.argsort(idx,   dim=0)            # inverse permutation
                    mask = (ranks < fixed_nn_neigh)                       # (n_obs, n_obs), bool
                    kernel_fixed_bw = kernel_fixed_bw * mask # (n_obs, n_obs, n_bw)
                    kernel_fixed_bw = kernel_fixed_bw.masked_fill(kernel_fixed_bw < min_wt, 0.0)
                    kernel_all[ str(other_slice) + '_' + str(currslice_idx)  ] = (kernel_fixed_bw / (kernel_fixed_bw.sum(dim=0, keepdim=True) + 0.000001)).cpu().numpy()           
    return kernel_all


def impute_slices(Palign,  slice_list, slice_order, num_mid=4, chunk_size = 500, key_name='linear_0'):
   
    num_sim  = num_mid + 1 # int((num_mid/2) + 1)
    in_betweens = (torch.arange(num_sim, device=device,dtype=float)[1:]) # .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    in_betweens = in_betweens/num_sim
    # adj_wts_extra = {}
    extra_slices = {}

    #print(in_betweens)

    for entry_tgt in range(len(slice_order)):
        tgt_slice = slice_order[entry_tgt]
        X_tgt = torch.tensor(slice_list[entry_tgt].X).to(device)
        nspots_tgt, ngenes = X_tgt.shape

        for entry_src in range(len(slice_order)):
            if entry_src not in [entry_tgt-1, entry_tgt+1]: # src must be 1 above or below tgt 
                continue
            print(entry_tgt,entry_src)
            src_slice = slice_order[entry_src]
            X_src = torch.tensor(slice_list[entry_src].X).to(device)
            nspots_src = X_src.shape[0]

            if tgt_slice == src_slice:
                continue
                
            Palign_curr = torch.tensor(Palign[(str(src_slice),str(tgt_slice))],device=device).unsqueeze(-1)
            
            for i in range(0,num_mid):
                mid_slice = torch.zeros((nspots_tgt, ngenes),device=device, dtype=X_tgt.dtype)
                
                for start in range(0, nspots_src, chunk_size):
                    end = min(start + chunk_size, nspots_src)
                    P_chunk     = Palign_curr[start:end, :, :]            # (chunk, n_tgt, 1)
                    src_chunk   = X_src[start:end, :]            # (chunk, n_genes)
                    diff_chunk  = X_tgt.unsqueeze(0) - src_chunk.unsqueeze(1) # slice_diff[start:end, :, :]            # (chunk, n_tgt, n_genes)
                    interp = src_chunk.unsqueeze(1) + in_betweens[i] * diff_chunk # (chunk, 1, n_genes) +  (chunk, n_tgt, n_genes)
                    weighted = P_chunk * interp # (chunk, n_tgt, 1)  (chunk, n_tgt, n_genes) -> (chunk, n_tgt, n_genes)
                    mid_slice += weighted.sum(dim=0) 

                noise = torch.normal(mean=0.0, std=0.1, size=mid_slice.shape,device=device, dtype=mid_slice.dtype)    

                
                extra_slices[(str(src_slice),str(tgt_slice), str(i))] = mid_slice.add(noise).cpu().numpy()
                # adj_wts_extra[(str(src_slice),str(tgt_slice), str(i))] = (adj_wts[entry_src,entry_src] + in_betweens[i]*(adj_wts[entry_tgt,entry_src] - adj_wts[entry_src,entry_src])).detach().cpu()
    return extra_slices# , adj_wts_extra