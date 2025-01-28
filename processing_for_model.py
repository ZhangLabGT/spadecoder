from importing_modules import *

###################### OLD CODE BEGINS ###############
# this using median(spa_dist) but leave out for now 
def get_gauss_kernel_wt_v2(adata_spa,spatial_key='spatial',min_wt=0.0001, bandwidth=1,n_neigh=10,recompute=False):
    
    ## STEPS:
    # a. get spatial distances
    # b. kernelize spatial distances only 
    
    # consider adding scRNA distances later in a weighted neighborhood graph (citeseq seurat paper)
    
    # using 
    if adata_spa.shape[0] < (n_neigh+1):
        n_neigh = adata_spa.shape[0] - 1

    spa_dist, spa_NNconn = get_intrasample_spatial_dist(adata_spa, spatial_key=spatial_key,n_neigh=n_neigh,recompute=recompute)
    spa_dist = spa_dist/np.max(spa_dist)  # scale to [0,1]


    mdis = 0.5 * bandwidth * np.median(spa_dist)

    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)

    kernel_wt = np.multiply(kernel_wt,spa_NNconn) # restrict to NN only 

    kernel_wt[kernel_wt<min_wt]=0.0

    np.fill_diagonal(kernel_wt, 1.0)

    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))
    
    # print(spa_dist)
    ## differences from cespgrn 
    # not sure why this is the form - I changed sligtly 
    #mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    # K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    
    # used the form from GPSA in kernel
    
    
    # make the wt of each column sum to 1 
    # spa_dist[spa_dist==0] = np.nan
    
    # kernel_wt = np.exp(-0.5*((spa_dist-np.nanmean(spa_dist))**2)/np.nanvar(spa_dist))
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    
    # replace nan's with 0 
    #print((np.isnan(kernel_wt)).astype(int).sum())
    # kernel_wt[np.isnan(kernel_wt)] = 0.0
    
    # # set all wts less than min_wt to 0
    # # kernel_wt[kernel_wt]
    # kernel_wt[kernel_wt<min_wt]=0.0
    
    # # set diagonal entries to 1 
    # np.fill_diagonal(kernel_wt, 1.0)

    # # normalize s.t. every columns sums to 1
    # kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))
    
    # print(kernel_wt)
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    return kernel_wt
   
def get_gauss_kernel_wt_orig(adata_spa,spatial_key='spatial',min_wt=0.0001,n_neigh=10,recompute=False):
    
    ## STEPS:
    # a. get spatial distances
    # b. kernelize spatial distances only 
    
    # consider adding scRNA distances later in a weighted neighborhood graph (citeseq seurat paper)
    
    # using 
    if adata_spa.shape[0] < (n_neigh+1):
        n_neigh = adata_spa.shape[0] - 1

    spa_dist = get_intrasample_spatial_dist_orig(adata_spa, spatial_key=spatial_key,n_neigh=n_neigh,recompute=recompute)
    
    # print(spa_dist)
    ## differences from cespgrn 
    # not sure why this is the form - I changed sligtly 
    #mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    # K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    
    # used the form from GPSA in kernel
    
    
    # make the wt of each column sum to 1 
    spa_dist[spa_dist==0] = np.nan
    
    kernel_wt = np.exp(-0.5*((spa_dist-np.nanmean(spa_dist))**2)/np.nanvar(spa_dist))
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    
    # replace nan's with 0 
    #print((np.isnan(kernel_wt)).astype(int).sum())
    kernel_wt[np.isnan(kernel_wt)] = 0.0
    
    # set all wts less than min_wt to 0
    # kernel_wt[kernel_wt]
    kernel_wt[kernel_wt<min_wt]=0.0
    
    # set diagonal entries to 1 
    np.fill_diagonal(kernel_wt, 1.0)

    # normalize s.t. every columns sums to 1
    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))
    
    # print(kernel_wt)
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    return kernel_wt

def get_intrasample_spatial_dist_orig(adata_spatial, spatial_key='spatial',n_neigh=10,recompute=False):
    ### computing spatial distances between cells in same sample 
    # distance between spatial 
    # spatial_sample is  anndata objects with spaital coordinates in adata.obsm[spatial_key]
    # cost_sp = distance_matrix(adata_spatial.obsm[spatial_key], adata_spatial.obsm[spatial_key]) # euclidean distance
    
    # if n_neigh is not None:
    #     # truncate to neighbors
    #     sq.gr.spatial_neighbors(adata_spatial, coord_type="generic",n_neighs=10) 
    if adata_spatial.shape[0] < (n_neigh+1):
        n_neigh = adata_spatial.shape[0] - 1

    if (recompute or ('spatial_connectivities'  not in adata_spatial.obsp.keys()) or (adata_spatial.uns['spatial_neighbors']['params']['n_neighbors']!=n_neigh)):
        sq.gr.spatial_neighbors(adata_spatial, spatial_key=spatial_key,coord_type="generic",n_neighs=n_neigh) 
    
    # this is not symmetric, every column sums to 10 (since 10neighbors) but every row doesnt
    spa_dist = adata_spatial.obsp['spatial_distances'].toarray() 

    # spa_dist = spa_dist / np.max(spa_dist) # rescale to [0,1]
    

    # differences from cespgrn
    # 1. didnt make k-NN symmetric (that seems odd )
    # 2. didn't calculate and use shortest path distances but directly used only the k-NN 
    return spa_dist

############### OLD CODE ENDS ###########################################

def get_gene_meanstd_expr_from_scrna_h5ad(filepath, cluster_key, layer='raw', 
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
    
    class_mean_per_gene = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = adata.X[idx]
        class_mean_per_gene[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
        
    # get per-class variance, for each gene 
    class_std_per_gene = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )


    for group, idx in grouped.indices.items():
        X = adata.X[idx]
        class_std_per_gene[group] = np.ravel(X.toarray().std(axis=0, dtype=np.float64))


    


    return adata_scrna, class_mean_per_gene, class_std_per_gene




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

def get_metacells(filepath, cluster_key, layer=None, 
                                  genes_to_use='all',numcells_inmeta=50, scrna_var_key='index'):
    
    adata = sc.read(filepath)


    if layer == 'raw':
        adata = adata.raw.to_adata()
    else:
        adata = adata.copy()

    if scrna_var_key != 'index':
        var_df = sc.get.var_df(adata,keys=list(adata.var.columns))
        var_df['old_idx'] = var_df.index
        var_df.set_index(scrna_var_key,inplace=True)
        adata = anndata.AnnData(X=adata.X, obs=adata.obs, var=var_df,obsm=adata.obsm, varm=adata.varm)

    if genes_to_use != 'all':
        # subset to genes intersecting with selected genes, possibly HVGs or those in ST 
        adata  = adata[:,list(set(adata.var.index).intersection(set(genes_to_use)))]

    
    meta_ct = []
    meta_identity = []
    ctypes = list(adata.obs[cluster_key].cat.categories)
    total_metas = 0
    for clus_entry in ctypes:
        adata_tmp = adata[adata.obs[cluster_key]==clus_entry]
        ncells_tmp = adata_tmp.shape[0]
        num_metas_tmp = math.ceil(ncells_tmp/numcells_inmeta)
        kmeans = KMeans(n_clusters=num_metas_tmp, random_state=0, n_init=10 ).fit(adata_tmp.X)
        tmp_df = pd.DataFrame(kmeans.cluster_centers_)
        tmp_meta_iden = pd.DataFrame(0, index=range(total_metas, total_metas+kmeans.cluster_centers_.shape[0]),columns=ctypes)
        tmp_meta_iden[clus_entry] = 1.0/kmeans.cluster_centers_.shape[0]

        total_metas = total_metas + kmeans.cluster_centers_.shape[0]

        meta_identity.append(tmp_meta_iden)
        meta_ct.append(tmp_df)

    meta_ct_all = pd.concat(meta_ct)
    meta_identity_all = pd.concat(meta_identity)

    meta_ct_all.columns = adata.var.index

    return adata, meta_ct_all, meta_identity_all


def get_ct_props_in_ref(adata_scrna, B,  ct_key="Cell class (determined from clustering of all cells)"):
    # input AnnData object
    # input B which contains cell-types we want to get proportions for in columns 
    
    adata_scrna = adata_scrna[adata_scrna.obs[ct_key].isin(B.columns),].copy()

    
    return adata_scrna.obs[ct_key].value_counts() / adata_scrna.shape[0]


def get_ct_props_in_ref_subset(adata_scrna, B, ct_key="Cell class (determined from clustering of all cells)"):
    # input AnnData object
    # input B which contains cell-types we want to get proportions for in columns 
    
    
    # adata_scrna = adata_scrna[adata_scrna.obs[ct_key].isin(B.columns),list(B.index)].copy()
    
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
    # spa_dist = spa_dist / np.max(spa_dist) # rescale to [0,1]
    # spadist = symmetric, spa_NNconn - not symmetric
    
    spadist = squareform(pdist(adata_spatial.obsm[spatial_key])) # distance (spatial) between every pari of cells

    # october 2024 debug update
    # print(np.allclose(spadist, spadist.T)) # check if symmetric
    # print(spa_NNconn.sum(axis=0)) # sums to k (for k-NN) 
    # print(spa_NNconn.sum(axis=1)) # does not sum to k

    # differences from cespgrn
    # 1. didnt make k-NN symmetric (that seems odd )
    # 2. didn't calculate and use shortest path distances but directly used only the k-NN 
    return spadist, spa_NNconn
 
def get_spatial_gauss_kernel_wt(adata_spa,spatial_key='spatial',min_wt=0.0, 
                        bandwidth=0.1,n_neigh=10,recompute=False,
                        nn_only=True):
    
    ## STEPS:
    # a. get spatial distances
    # b. kernelize spatial distances only 
    
    # consider adding scRNA distances later in a weighted neighborhood graph (citeseq seurat paper)
    
    # using 
    if adata_spa.shape[0] < (n_neigh+1):
        n_neigh = adata_spa.shape[0] - 1
    
    spa_dist, spa_NNconn = get_intrasample_spatial_dist(adata_spa, spatial_key=spatial_key,n_neigh=n_neigh,nn_only=nn_only,recompute=recompute)
    spa_dist = spa_dist/np.max(spa_dist)  # scale to [0,1]


    # mdis = 0.5 * bandwidth * np.median(spa_dist)
    mdis = 2*bandwidth*bandwidth

    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)

    if nn_only:
        kernel_wt = np.multiply(kernel_wt,spa_NNconn) # restrict to NN only 

    kernel_wt[kernel_wt<min_wt]=0.0

    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0)) # movd to before the next line in APR 2024

    np.fill_diagonal(kernel_wt, 1.0) # now the sum for each column should be 2


    return kernel_wt, spa_NNconn

def get_gauss_kernel_wt_v4(adata_spa,spatial_key='spatial',nn_only=True,min_wt=0.0001, bandwidth=1,n_neigh=10,recompute=False):
    
    ## STEPS:
    # a. get spatial distances
    # b. kernelize spatial distances only 
    
    # consider adding scRNA distances later in a weighted neighborhood graph (citeseq seurat paper)
    
    # using 
    if adata_spa.shape[0] < (n_neigh+1):
        n_neigh = adata_spa.shape[0] - 1

    spa_dist, spa_NNconn = get_intrasample_spatial_dist(adata_spa, spatial_key=spatial_key,n_neigh=n_neigh,recompute=recompute)
    spa_dist = spa_dist/np.max(spa_dist)  # scale to [0,1]


    mdis = 0.5 * bandwidth * np.median(spa_dist)

    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)

    if nn_only:
        kernel_wt = np.multiply(kernel_wt,spa_NNconn) # restrict to NN only 

    
    kernel_wt[kernel_wt<min_wt]=0.0

    # putting this here is compatible with the v1 version of spadecoder
    # as used in get_gauss_kernel_wt_v2

    np.fill_diagonal(kernel_wt, 1.0) # uncomment for previous verions

    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))


    # np.fill_diagonal(kernel_wt, 1.0) # now sums to 2 
    
    # print(kernel_wt)
    # print(spa_dist)
    ## differences from cespgrn 
    # not sure why this is the form - I changed sligtly 
    #mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    # K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    
    # used the form from GPSA in kernel
    
    
    # make the wt of each column sum to 1 
    # spa_dist[spa_dist==0] = np.nan
    
    # kernel_wt = np.exp(-0.5*((spa_dist-np.nanmean(spa_dist))**2)/np.nanvar(spa_dist))
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    
    # replace nan's with 0 
    #print((np.isnan(kernel_wt)).astype(int).sum())
    # kernel_wt[np.isnan(kernel_wt)] = 0.0
    
    # # set all wts less than min_wt to 0
    # # kernel_wt[kernel_wt]
    # kernel_wt[kernel_wt<min_wt]=0.0
    
    # # set diagonal entries to 1 
    # np.fill_diagonal(kernel_wt, 1.0)

    # # normalize s.t. every columns sums to 1
    # kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))
    
    # print(kernel_wt)
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    return kernel_wt, spa_NNconn


def get_kernel_wt_play(adata_spa,spatial_key='spatial',nn_only=True,min_wt=0.0001, bandwidth=1,n_neigh=10,recompute=False):
    
    ## STEPS:
    # a. get spatial distances
    # b. kernelize spatial distances only 
    
    # consider adding scRNA distances later in a weighted neighborhood graph (citeseq seurat paper)
    
    # using 
    if adata_spa.shape[0] < (n_neigh+1):
        n_neigh = adata_spa.shape[0] - 1

    spa_dist, spa_NNconn = get_intrasample_spatial_dist(adata_spa, spatial_key=spatial_key,n_neigh=n_neigh,recompute=recompute)
    spa_dist = spa_dist/np.max(spa_dist)  # scale to [0,1]


    mdis = 0.5 * bandwidth * np.median(spa_dist)

    kernel_wt  = np.exp(-(spa_dist ** 2)/mdis)

    if nn_only:
        kernel_wt = np.multiply(kernel_wt,spa_NNconn) # restrict to NN only 

    kernel_wt[kernel_wt<min_wt]=0.0

    # putting this here is compatible with the v1 version of spadecoder
    # as used in get_gauss_kernel_wt_v2

    np.fill_diagonal(kernel_wt, 1.0) # uncomment for previous verions

    kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))


    # np.fill_diagonal(kernel_wt, 1.0) # now sums to 2 
    
    # print(kernel_wt)
    # print(spa_dist)
    ## differences from cespgrn 
    # not sure why this is the form - I changed sligtly 
    #mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    # K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    
    # used the form from GPSA in kernel
    
    
    # make the wt of each column sum to 1 
    # spa_dist[spa_dist==0] = np.nan
    
    # kernel_wt = np.exp(-0.5*((spa_dist-np.nanmean(spa_dist))**2)/np.nanvar(spa_dist))
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    
    # replace nan's with 0 
    #print((np.isnan(kernel_wt)).astype(int).sum())
    # kernel_wt[np.isnan(kernel_wt)] = 0.0
    
    # # set all wts less than min_wt to 0
    # # kernel_wt[kernel_wt]
    # kernel_wt[kernel_wt<min_wt]=0.0
    
    # # set diagonal entries to 1 
    # np.fill_diagonal(kernel_wt, 1.0)

    # # normalize s.t. every columns sums to 1
    # kernel_wt = (kernel_wt/kernel_wt.sum(axis=0))
    
    # print(kernel_wt)
    # print(np.nanmin(kernel_wt),np.nanmax(kernel_wt))
    return kernel_wt, spa_NNconn



def combine_transcr_spatial_kernel(adata_spatial,spatial_key='spatial',min_wt=0.0, bandwidth=0.1,
                                   n_spatial_neigh=10,recompute=False,nn_only=True,
                                   n_transcr_neigh=15,n_pcs=20,
                                   kernel_combo_method='M1',
                                   weight_transcr=1,# only applies to M2
                                   weight_spatial=1):
    # try weighting by both transcr and spatial - M2
    # try restricting transcrip kernel to those in spatial proximity - M1
    if adata_spatial.shape[0] < (n_spatial_neigh+1):
        n_spatial_neigh = adata_spatial.shape[0] - 1

    if adata_spatial.shape[0] < (n_transcr_neigh+1):
        n_transcr_neigh = adata_spatial.shape[0] - 1

    # this is used in spadecoder_closedform_v2
    # spa_kernel_wt, spa_NNconn = get_spatial_gauss_kernel_wt(adata_spatial,spatial_key=spatial_key,min_wt=min_wt,
    #                     bandwidth=bandwidth,n_neigh=n_spatial_neigh,recompute=recompute,nn_only=nn_only)

    spa_kernel_wt,spa_NNconn = get_gauss_kernel_wt_v4(adata_spatial,spatial_key=spatial_key,min_wt=min_wt,
                        bandwidth=bandwidth,n_neigh=n_spatial_neigh,recompute=recompute,nn_only=nn_only)

    if kernel_combo_method == 'M2' and weight_transcr == 0.0:
        transc_kernel_wt = 0.0
    else:
        transc_kernel_wt = get_transcr_gauss_kernel_wt(adata_spatial, recompute=recompute,n_neigh=n_transcr_neigh,n_pcs=n_pcs)
    
    if kernel_combo_method == 'M1':
        # restricting transcrip kernel to those in spatial proximity
        # make spatial conn matrix symmetric 
        spa_conn_symm = ((spa_NNconn +  spa_NNconn.T).astype(bool).astype(int)) + np.eye(adata_spatial.shape[0])
        net_kernel = np.multiply(transc_kernel_wt,spa_conn_symm)
    elif kernel_combo_method == 'M2':
        # try weighting by both transcr and spatial - M2
        net_kernel = weight_transcr*transc_kernel_wt + weight_spatial*spa_kernel_wt # weight by both transcr and spatia
    else:
        print("invalid kernerl combine method")
    return net_kernel

def get_intrasample_transcr_dist(adata_spatial, recompute=False,n_neigh=15,n_pcs=20):
    
    # transcr_NNconn = np.ones((adata_spatial.shape[0],adata_spatial.shape[0]))
    if (recompute or ('connectivities'  not in adata_spatial.obsp.keys()) or 
            (adata_spatial.uns['neighbors']['params']['n_neighbors']!=n_neigh) or 
            (adata_spatial.uns['neighbors']['params']['n_pcs']!=n_pcs)): # recompute 
            sc.pp.neighbors(adata_spatial, n_neighbors=n_neigh,n_pcs=n_pcs) 
        

    transcr_NNconn = adata_spatial.obsp['connectivities'].toarray() # this is the weighted connectivity, not the same as dist 
    transcr_NNconn = transcr_NNconn + np.eye(adata_spatial.shape[0]) # this means that the sum total is 2
    
    transcr_NN_dist = adata_spatial.obsp['distances'].toarray() # these are the actual distances

    return transcr_NNconn,transcr_NN_dist


def get_transcr_gauss_kernel_wt(adata_spatial, recompute=False,n_neigh=15,n_pcs=20):
    # use scanpy NN connectivity for now, modify later
   

    transcr_NNconn, transcr_NN_dist = get_intrasample_transcr_dist(adata_spatial, n_pcs=n_pcs,
                                                        n_neigh=n_neigh,recompute=recompute)
    # unlike in the spatial case, transcr_NNconn is symmetric (I've used scanpy's NN implementation) 
    # renormalize so each row ( or column, since it's symmetric ) sums to 1 - try later 

    # kernel_wt = (transcr_NNconn/transcr_NNconn.sum(axis=0)) 
    

    return transcr_NNconn