# _tissue denotes simulating multiple slices via successive warping 

from .importing_modules import *



def make_spot(adata_sc,N=50,spatial_key='spatial',spot_key='spot_number',
              spa_celltype_key='Types',wts=None):
    

    # rescale X and Y coordinates to [0,1] 


    # if no weights ( to account for cell size, then set to 1 )
    if wts is None:
        wts = np.ones(adata_sc.shape[0])
        wts = wts[:, np.newaxis]
    
    step_sz = math.sqrt(N/adata_sc.shape[0])
    
    # start at bottom left (0,0)
    x_curr = step_sz
    x_prev = -0.000001
    
    y_curr = step_sz
    y_prev = -0.000001
    
    curr_spot_num = 0
    spot_numbers = pd.DataFrame([-1]*adata_sc.shape[0],index=range(adata_sc.shape[0]),columns=[spot_key])
    # cell_idx = 0
    
    spas = adata_sc.obsm[spatial_key]
    spas = (spas - spas.min(axis=0))/(spas.max(axis=0)  - spas.min(axis=0))
    spa_coord = pd.DataFrame(spas)

    # rescale to [0,1]
    
    
    # get spot-expression matrix 
    df_spot = pd.DataFrame(0,index=adata_sc.var.index,columns=[])
    
    # unsparsify if necessary  
    if sp.sparse.issparse(adata_sc.X):
        adata_sc.X = adata_sc.X.toarray()
        
    weighted_expr = adata_sc.X * wts  # weight the expression 
    
    
    # create cell-type composition matrix 
    spot_ct = pd.DataFrame(0,index=list(adata_sc.obs[spa_celltype_key].cat.categories),columns=[])
    one_hot_orig = pd.get_dummies(adata_sc.obs[spa_celltype_key],dtype=float)
    weighted_ct = one_hot_orig * wts
    weighted_ct.index = list(range(adata_sc.shape[0]))
    weighted_ct = weighted_ct[list(adata_sc.obs[spa_celltype_key].cat.categories)].copy() # to make sure order is same
    
    
    # create spot-spatial coordinates - 2D matrix
    spot_spa_coord = pd.DataFrame(0,index=[0,1],columns=[])
    weighted_spa_coord = spa_coord * wts  # since we're finding center-of-mass
    
    
    while y_prev < 1:
        y_curr = min(y_curr,1)
        while x_prev <1:
            x_curr = min(x_curr,1)
            
            spa_tmp = spa_coord[(spa_coord[0] <= x_curr) & (spa_coord[0] > x_prev) & (spa_coord[1] <= y_curr) & (spa_coord[1] > y_prev )]
            
            
            if spa_tmp.shape[0] > 0:
                
                # print(spa_tmp.shape[0],curr_spot_num,x_prev, x_curr, y_prev, y_curr)
                
                spot_numbers.loc[list(spa_tmp.index),spot_key] = curr_spot_num
                
                # get weighted average expression of cells in a spot 
                # df_spot[curr_spot_num] = weighted_expr[spa_tmp.index,].mean(axis=0)
                df_spot[curr_spot_num] = weighted_expr[spa_tmp.index,].sum(axis=0) # changed to sum from mean on Nov 18 2024 
                
                # get cell-type composition of cells in the spot 
                    
                spot_ct[curr_spot_num] = weighted_ct.loc[list(spa_tmp.index),].sum(axis=0) # weighted_ct[spa_tmp.index,].sum(axis=0)
                
                # get new spot-location co-ordinate (center of mass)
                spot_spa_coord[curr_spot_num] = weighted_spa_coord.loc[list(spa_tmp.index),].mean(axis=0)
                
                curr_spot_num = curr_spot_num + 1
            
            x_prev = x_curr
            x_curr = x_curr + step_sz
        
        y_prev = y_curr
        y_curr = y_curr + step_sz
        
        x_prev = -0.000001
        x_curr = step_sz
    
    spot_numbers.index = adata_sc.obs.index
    
    # delete existing if it exists
    if spot_key in adata_sc.obs.columns:
        del adata_sc.obs[spot_key]
        
    adata_sc.obs = adata_sc.obs.join(spot_numbers)
    # print("spot gen " + str(adata_sc.obs[spot_key].max()))
    
    # create adata_spot object 
    
    adata_spot = anndata.AnnData(X=df_spot.T)
    adata_spot.obsm[spatial_key] = np.array(spot_spa_coord.T)
    adata_spot.obs = spot_ct.T
    
    adata_spot.obs.index = adata_spot.obs.index.astype(str)
    adata_spot.var.index = adata_spot.var.index.astype(str)
    
    return adata_sc,adata_spot



def cell_swap_for_allsims_tissue_final(adata_sc, adata_spot, 
                                    nswaps_nbd=5, # slice_num=1, 
                                    spatial_key='spatial', n_neigh=10,
                                    spot_key = 'spot_number', 
                                    swap_seed=1,
                                    spa_celltype_key='Types',
                                    wts=None):
    # here I select a fixed ( by parameter ) number of cells from each neighborhood of spots and shuffle their labels
    # parameters: nbd size (2-5), n_neigh (10) # cells 
    
    spot_info = adata_sc.obs[spot_key]
    spot_ids = list(adata_spot.obs.index) 
    
    # calculate updated weighted expression and spot composition
    if wts is None:
        wts = np.ones(adata_sc.shape[0])
        wts = wts[:, np.newaxis]
        
    # unsparsify if necessary  
    if sp.sparse.issparse(adata_sc.X):
        adata_sc.X = adata_sc.X.toarray()
    
    # cell by expression matrix 
    weighted_expr = pd.DataFrame(adata_sc.X * wts,index=adata_sc.obs.index,columns=adata_sc.var.index)  # weight the expression 

    # cell by cell-type matrix  - cell-type composition matrix 
    one_hot_orig = pd.get_dummies(adata_sc.obs[spa_celltype_key],dtype=float)
    weighted_ct = one_hot_orig * wts
    weighted_ct.index = adata_sc.obs.index # list(range(adata_sc.shape[0]))
    weighted_ct = weighted_ct[list(adata_sc.obs[spa_celltype_key].cat.categories)].copy() # to make sure order is same
    
    # cell-type by spot matrix 
    adata_sc.obs[spot_key] = adata_sc.obs[spot_key].astype(str)
    df_encoded = pd.get_dummies(adata_sc.obs[spot_key], columns=[spot_key]).astype(float) # cell by spot matrix 
    df_encoded = df_encoded[spot_ids].copy()
    spot_ct = weighted_ct.T @ df_encoded # get weighted cell-type by spot matric 
   
    df_spot = weighted_expr.T @ df_encoded  # gene by spot weighted expr matrix, # gene by cell x cell by spot  
   
    # get spatial neighbors 
    # this is not symmetric, every column sums to 10 (since 10neighbors) but every row doesnt
    if adata_spot.shape[0] <= n_neigh:
        n_neigh = adata_spot.shape[0] - 1
    sq.gr.spatial_neighbors(adata_spot, spatial_key=spatial_key,coord_type="generic",n_neighs=n_neigh) 
    spa_NNconn = adata_spot.obsp['spatial_connectivities'].toarray() # 10NN connectivity 
    np.fill_diagonal(spa_NNconn, 1.0) # make sure spot itself is included in its neighborhood
    spa_NNconn = pd.DataFrame(spa_NNconn,index=spot_ids,columns=spot_ids)
    # cell-swapping strategey 
    # for each spot
    # get nbd 
    # get all cells in spots 
    # randomly pick a subset 
    # shuffle spot labels 
    swap_seed = swap_seed*500

    # for debugging 
    all_spots_swapped = []
    spot_info_orig = spot_info.copy()
    
    for spot_idx in spa_NNconn.columns:
        # get spots, nbd, cells in nbd 
        cells_in_nbd = list(adata_sc[adata_sc.obs[spot_key].isin(list(spa_NNconn[spa_NNconn[spot_idx]==1][spot_idx].index))].obs.index)
        # pick random cells & shuffle spot label 
        rng = np.random.default_rng(seed=swap_seed)

        if nswaps_nbd > len(cells_in_nbd):
            nswaps_nbd_curr = len(cells_in_nbd)
        else:
            nswaps_nbd_curr = nswaps_nbd
        cells_to_swap = rng.choice(cells_in_nbd, nswaps_nbd_curr, replace=False)
        selected_spots = spot_info[cells_to_swap]
        swap_seed = swap_seed + 1
        
        # for debugging
        selected_spots_orig = selected_spots.copy()
        # debugging ends 
        
        rng.shuffle(selected_spots) # swap ordering of spots
        
        # for debugging 
        all_spots_swapped.extend(list(selected_spots[selected_spots!=selected_spots_orig].values))
        # debugging ends 

        # Step 4: Update DataFrame with shuffled spots
        spot_info[cells_to_swap] = selected_spots

    # for debugging 
    num_swapped = spot_info[spot_info_orig!=spot_info].shape[0] # numcells swapped 
    total_spots = spot_info.shape[0]
    tmpa = spot_info[spot_info_orig != spot_info]
    tmpb = tmpa.value_counts()/(spot_info.value_counts()[tmpa.value_counts().index])
    # print(f"Cells swapped: {num_swapped}, Total Cells: {total_spots}, Percent: {round((100*float(num_swapped)/float(total_spots)),2)}, Number spots changed: {len(set(all_spots_swapped))}")
    # print(f"Max pct cells swapped in spot: {round((100*tmpb.max()),2)}, Min pct cells swapped in spot: {round((100*tmpb.min()),2)}, Median pct cells swapped in spot: {round((100*tmpb.median()),2)}")
    pct_cells_swapped = list((100*tmpb).values)
    pct_spots_changed = tmpa.value_counts().shape[0]/adata_spot.shape[0]

    # now make spots with the new assignment 
    adata_sc.obs[spot_key] = spot_info # update the spot positions
    adata_sc.obs[spot_key] = adata_sc.obs[spot_key].astype(str)
    # get spot_ct (spot weighted cell-type numbers) and df_spot (spot weighted expr)
    df_encoded = pd.get_dummies(adata_sc.obs[spot_key], columns=[spot_key]).astype(float) # cell by spot matrix 
    df_encoded = df_encoded[spot_ids].copy()
    df_spot = weighted_expr.T @ df_encoded  # gene by spot weighted expr matrix, # gene by cell x cell by spot  
    spot_ct = weighted_ct.T @ df_encoded # get weighted cell-type by spot matric 

    # update adata_spot.X, adata_spot.obs ( cell counts )
    adata_spot_swap = adata_spot.copy() # anndata.AnnData(X=np.array(df_spot.T))
    # print(adata_sc.shape,adata_spot.shape,df_spot.shape)
    # adata_spot_warp.obsm['spatial'] = np.array(spot_spa_coord.T)
    adata_spot_swap.X = np.array(df_spot.T)
    adata_spot_swap.obs = spot_ct.T
    
    adata_spot_swap.obs.index = adata_spot_swap.obs.index.astype(str)
    adata_spot_swap.var.index = adata_spot_swap.var.index.astype(str)
    
    return adata_sc, adata_spot_swap, pct_cells_swapped, pct_spots_changed
### new swapping ends ##################################



def sim_multiple_slices_allsims_tissue_forbatch_nowarp(adata_st,# types_of_spots=['linear','polar','gaussian'],
                        spa_celltype_key='Types',N=50, # save_spot_in_slices='spots_simulated.pickle',
                        save_sim_slices='warped_spots_simulated.pickle',wts=None,
                        num_tissueslices = 10,spot_base_key='spot_number',
                        spatial_base_key = 'spatial',
                        nswaps_nbd=5, n_neigh=10):
    
    # note that if we decide to swap too many spots, it won't make a difference as the maximum number 
    # will be swapped in this version of the code 

    adata_orig = {}
    adata_spot_swap = {}
    
    # parameters for seeds 
    seed_par = 23

    types_of_sims_curr = 'linear_0'
    
    # initialize 
    adata_orig[types_of_sims_curr] = {}
    adata_spot_swap[types_of_sims_curr] = {}
    for entry0 in range(num_tissueslices+1):
        adata_spot_swap[types_of_sims_curr][entry0] = {}
        adata_orig[types_of_sims_curr][entry0] = {}

    ######## MAKE SPOTS #########
    for slice_idx in adata_st.keys(): # real slices, 0th simulation which is same as original slice
        adata_orig[types_of_sims_curr][0][slice_idx], adata_spot_swap[types_of_sims_curr][0][slice_idx] = make_spot(adata_st[slice_idx],
                                                                                                                    N=N,
                                                                                                                    wts=wts, 
                                                                                                                    spa_celltype_key=spa_celltype_key,
                                                                                                                    spatial_key=spatial_base_key,
                                                                                                                    spot_key=spot_base_key)
    ######### SWAP TO GENERATE MULTIPLE SLICES ##############
    pct_spots_changed = []
    pct_cells_swapped = []
    for slice_idx in adata_st.keys(): # real slices 
        # cnt = 0    
        for entry0 in range(num_tissueslices): # simulated slices 
                seed_par += 1 
                adata_orig[types_of_sims_curr][entry0+1][slice_idx], adata_spot_swap[types_of_sims_curr][entry0+1][slice_idx], pct_cells_swapped_tmp, pct_spots_changed_tmp = cell_swap_for_allsims_tissue_final(adata_orig[types_of_sims_curr][entry0][slice_idx], # this is scRNA
                                                                            adata_spot_swap[types_of_sims_curr][entry0][slice_idx],
                                                                            # numcells_to_swap=numcells_to_swap,  # not used 
                                                                            nswaps_nbd=nswaps_nbd,
                                                                            spatial_key = spatial_base_key, 
                                                                            n_neigh = n_neigh,
                                                                            spot_key = spot_base_key, # 'spot_number' for the 1st, spot_key_warped after
                                                                            spa_celltype_key=spa_celltype_key,
                                                                            swap_seed=seed_par,
                                                                            wts=wts)  
                pct_cells_swapped.extend(pct_cells_swapped_tmp)
                pct_spots_changed.append(pct_spots_changed_tmp)

    with open(save_sim_slices, 'wb') as handle:
        pickle.dump(adata_spot_swap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pct_cells_swapped, pct_spots_changed # adata_spot