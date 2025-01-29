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
    
    
    # create adata_spot object 
    
    adata_spot = anndata.AnnData(X=df_spot.T)
    adata_spot.obsm[spatial_key] = np.array(spot_spa_coord.T)
    adata_spot.obs = spot_ct.T
    
    adata_spot.obs.index = adata_spot.obs.index.astype(str)
    adata_spot.var.index = adata_spot.var.index.astype(str)
    
    return adata_sc,adata_spot


def make_warp_linear(adata_spot,spatial_key='spatial',linear_slope_variance=0.1,
                     linear_intercept_variance=0.1,
                     noise_mean= np.zeros(2),
                     noise_var_coeff=0.0005,
                     linear_warp_slope_seed=1,
                     linear_warp_int_seed=1,
                     linear_noise_seed=1):
    
    n_spots = adata_spot.shape[0]
    n_genes = adata_spot.shape[1]
    
    rng = np.random.default_rng(seed=linear_warp_slope_seed)

    curr_slopes = rng.uniform(
            low=1 - linear_slope_variance,
            high=1 + linear_slope_variance,
            size=2,
        )
    
    rng = np.random.default_rng(seed=linear_warp_int_seed)

    curr_intercepts = rng.uniform(
            low=-linear_intercept_variance,
            high=linear_intercept_variance,
            size=2,
        )
    
    
    
    X_curr_view_warped = adata_spot.obsm[spatial_key] * curr_slopes + curr_intercepts
    
    rng = np.random.default_rng(seed=linear_noise_seed)
    # print(noise_mean)
    add_noise = rng.multivariate_normal(
             mean=noise_mean, # np.mean(adata_spot.obsm[spatial_key], axis=0),
             cov=noise_var_coeff*np.eye(2), size=adata_spot.shape[0]) # np.cov(adata_spot.obsm[spatial_key],rowvar=False), 
    
    X_curr_view_warped = X_curr_view_warped + add_noise
    
    
    # shift co-ordinates to 0,1 since that's the range of adata.obsm['spatial']
    data_min = np.min(X_curr_view_warped,axis=0)
    data_max = np.max(X_curr_view_warped,axis=0)
    X_curr_view_warped = np.zeros(2) + (X_curr_view_warped - data_min) * (np.ones(2) - np.zeros(2)) / (data_max - data_min)
    
    # print(adata_spot.shape,X_curr_view_warped.shape)
    return X_curr_view_warped

def cell_swap_for_allsims_tissue(adata_sc, adata_spot, 
                                    numcells_to_swap=2, # slice_num=1, 
                                    spot_key = 'spot_number', 
                                    swap_seed=1,
                                    spa_celltype_key='Types',
                                    wts=None):

    rng = np.random.default_rng(seed=swap_seed)
    spot_info = adata_sc.obs[spot_key]
    cells_in_spots = adata_sc.obs[spot_key].value_counts()
    spot_ids = list(cells_in_spots.index)
    spot_ids.sort() # to facilitate swapping between adjacent spot ids
    
    spot_id_idx = 0
    
    # spot_swapped = pd.DataFrame([-1]*adata_sc.shape[0],columns=list(adata_sc.index),index=spot_key + '_swapped')
    
    # bug fix 
    adata_sc = adata_sc.copy()
    obs = pd.DataFrame(adata_sc.obs)
    obs[spot_key+ '_swapped'] = obs[spot_key].copy()
    
    # adata.obs = obs 
    adata_sc.obs = obs #[spot_key+'_swapped'] = adata_sc.obs[spot_key].copy()

    # calculate updated weighted expression and spot composition
    if wts is None:
        wts = np.ones(adata_sc.shape[0])
        wts = wts[:, np.newaxis]
        
    # get spot-expression matrix 
    df_spot = pd.DataFrame(-1.0,index=adata_sc.var.index,columns=[])
    
    # unsparsify if necessary  
    if sp.sparse.issparse(adata_sc.X):
        adata_sc.X = adata_sc.X.toarray()
        
    weighted_expr = pd.DataFrame(adata_sc.X * wts,index=adata_sc.obs.index,columns=adata_sc.var.index)  # weight the expression 
    
    
    # create cell-type composition matrix 
    spot_ct = pd.DataFrame(0,index=list(adata_sc.obs[spa_celltype_key].cat.categories),columns=[])
    one_hot_orig = pd.get_dummies(adata_sc.obs[spa_celltype_key],dtype=float)
    weighted_ct = one_hot_orig * wts
    weighted_ct.index = adata_sc.obs.index # list(range(adata_sc.shape[0]))
    weighted_ct = weighted_ct[list(adata_sc.obs[spa_celltype_key].cat.categories)].copy() # to make sure order is same
    
    
    
    while spot_id_idx <= (len(spot_ids) -2):
        
        num_cells_spot_1 = cells_in_spots[spot_ids[spot_id_idx]]
        num_cells_spot_2 = cells_in_spots[spot_ids[spot_id_idx+1]]

        # print("num cells in spot " + str(num_cells_spot_1) + "_" + str(num_cells_spot_2) )
        if numcells_to_swap > min(num_cells_spot_1,num_cells_spot_2):
            num_swaps = min(num_cells_spot_1,num_cells_spot_2)
        else:
            num_swaps = numcells_to_swap

        # num_swaps = math.floor(min(numcells_to_swap,num_cells_spot_1/5,num_cells_spot_2/5))
        
        # pick num_swaps cells 
        swap_1 = rng.integers(low=0, high=num_cells_spot_1, size=num_swaps)   # random.sample(range(0,num_cells_spot_1), num_swaps)
        swap_2 = rng.integers(low=0, high=num_cells_spot_2, size=num_swaps) # random.sample(range(0,num_cells_spot_2), num_swaps)
        
        # update spot_numbers in spot_number_swapped
        swap_cell_names_1 = adata_sc[adata_sc.obs[spot_key] == spot_id_idx].obs.index[swap_1]
        swap_cell_names_2 = adata_sc[adata_sc.obs[spot_key] == spot_id_idx+1].obs.index[swap_2]
        
        adata_sc.obs.loc[swap_cell_names_1,spot_key+ '_swapped'] = spot_ids[spot_id_idx+1]
        adata_sc.obs.loc[swap_cell_names_2,spot_key+'_swapped'] = spot_ids[spot_id_idx]
        
        all_idx_for_fst_spot = list(adata_sc[adata_sc.obs[spot_key+'_swapped']==spot_ids[spot_id_idx]].obs.index)
        all_idx_for_scnd_spot = list(adata_sc[adata_sc.obs[spot_key+'_swapped']==spot_ids[spot_id_idx+1]].obs.index)
        
        spot_ct[spot_ids[spot_id_idx]] = weighted_ct.loc[all_idx_for_fst_spot,].sum(axis=0)
        spot_ct[spot_ids[spot_id_idx+1]] = weighted_ct.loc[all_idx_for_scnd_spot,].sum(axis=0)
        
        # df_spot[spot_ids[spot_id_idx]] = weighted_expr.loc[all_idx_for_fst_spot,].mean(axis=0)
        # df_spot[spot_ids[spot_id_idx+1]] = weighted_expr.loc[all_idx_for_scnd_spot,].mean(axis=0)
        
        df_spot[spot_ids[spot_id_idx]] = weighted_expr.loc[all_idx_for_fst_spot,].sum(axis=0) # replaced on Nov 18 2024 
        df_spot[spot_ids[spot_id_idx+1]] = weighted_expr.loc[all_idx_for_scnd_spot,].sum(axis=0) # replaced on Nov 18 2024 
        

        spot_id_idx = spot_id_idx + 2
        
    # copy last spot without a swap, if odd number of spots
    if adata_spot.shape[0]%2 != 0:
        df_spot[spot_ids[spot_id_idx]] = adata_spot.X[-1,:]
        spot_ct[spot_ids[spot_id_idx]] = adata_spot.obs.loc[adata_spot.obs.index[-1],]
    
    # update adata_spot.X, adata_spot.obs ( cell counts )
    adata_spot_swap = adata_spot.copy() # anndata.AnnData(X=np.array(df_spot.T))
    # print(adata_sc.shape,adata_spot.shape,df_spot.shape)
    # adata_spot_warp.obsm['spatial'] = np.array(spot_spa_coord.T)
    adata_spot_swap.X = np.array(df_spot.T)
    adata_spot_swap.obs = spot_ct.T
    
    adata_spot_swap.obs.index = adata_spot_swap.obs.index.astype(str)
    adata_spot_swap.var.index = adata_spot_swap.var.index.astype(str)
    
    # adata_sc contains the swapped IDs

    # move this so for the next round in the series of slices we'll start from spot number 
    adata_sc.obs[spot_key] = adata_sc.obs[spot_key + '_swapped'].copy()
    del adata_sc.obs[spot_key + '_swapped']

    # adata_spot_swap
    return adata_sc, adata_spot_swap

def make_warp_polar(adata_spot,polar_slope_variance=0.1,spatial_key='spatial',
                    noise_mean= np.zeros(2),noise_var_coeff=0.0005, polar_warp_seed=1,polar_noise_seed=1):
    # didn't really test by checking outputs and tallying with manually computed
    n_spots = adata_spot.shape[0]
    n_genes = adata_spot.shape[1]
    
    rng = np.random.default_rng(seed=polar_warp_seed)


    B = rng.uniform(
            low=-polar_slope_variance,
            high=polar_slope_variance,
            size=(2, 2)
    )

    polar_params = adata_spot.obsm[spatial_key] @ B
    
    r, theta = polar_params[:, 0], polar_params[:, 1]
    
    
    X_polar_warped = np.array(
            [
                adata_spot.obsm[spatial_key][:, 0] + r * np.cos(theta),
                adata_spot.obsm[spatial_key][:, 1] + r * np.sin(theta),
            ]
        ).T
    
    rng = np.random.default_rng(seed=polar_noise_seed)
    add_noise = rng.multivariate_normal(
            mean=noise_mean,
            cov=noise_var_coeff*np.eye(2),size=adata_spot.shape[0])
    
    X_polar_warped = X_polar_warped + add_noise
    
    data_min = np.min(X_polar_warped,axis=0)
    data_max = np.max(X_polar_warped,axis=0)
    X_polar_warped = np.zeros(2) + (X_polar_warped - data_min) * (np.ones(2) - np.zeros(2)) / (data_max - data_min)
    
    return X_polar_warped

def rbf_kernel_numpy(x, xp, kernel_params):
    # from GPSA
    output_scale = np.exp(kernel_params[0]) # 1 is default 
    # print(output_scale)
    lengthscales = np.exp(kernel_params[1:])# 1 is default 
    # print(lengthscales)
    # subtract every entry from every other entry
    diffs = np.expand_dims(x / lengthscales, 1) - np.expand_dims(xp / lengthscales, 0) 
    
    # print(np.expand_dims(x / lengthscales, 1).shape,np.expand_dims(xp / lengthscales, 0))
    # print(xp == x)
    
    # print(diffs.shape)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

def make_warp_gaussian(adata_spot,spatial_key='spatial',
                      # noise_variance=0.0,
    gauss_kernel_variance=1.0,
    gauss_kernel_lengthscale=1.0,
    # gauss_mean = 0.0,
    gauss_mean_slope=1.0,
    gauss_mean_intercept=0.0,
    noise_mean= np.zeros(2),noise_var_coeff=0.0005, 
    gauss_seed=1,gauss_noise_seed=1):
    
    # didn't test in too much detail
    n_spots = adata_spot.shape[0]
    n_spatial_dims = 2
    # n_genes = adata_spot.shape[1]
    
    # kernel_params_true = np.array([np.log(gauss_kernel_variance), np.log(gauss_kernel_lengthscale)])
    
    warp_kernel_params_true = np.array(
        [np.log(gauss_kernel_variance), np.log(gauss_kernel_lengthscale)]
    )
    
    X_curr_view_warped = adata_spot.obsm[spatial_key].copy()
    
    rng = np.random.default_rng(seed=gauss_seed)
    for ss in range(n_spatial_dims):
            X_curr_view_warped[:,ss] = rng.multivariate_normal(
                mean=adata_spot.obsm[spatial_key][:, ss] * gauss_mean_slope + gauss_mean_intercept,
                # I think there's a problem with this covariance
                cov=rbf_kernel_numpy(adata_spot.obsm[spatial_key], adata_spot.obsm[spatial_key], warp_kernel_params_true))
     
    rng = np.random.default_rng(seed=gauss_noise_seed)
    add_noise = rng.multivariate_normal(
            mean=noise_mean,
            cov=noise_var_coeff*np.eye(2), size=n_spots)
    
    X_curr_view_warped = X_curr_view_warped + add_noise
    
    data_min = np.min(X_curr_view_warped,axis=0)
    data_max = np.max(X_curr_view_warped,axis=0)
    X_curr_view_warped = np.zeros(n_spatial_dims) + (X_curr_view_warped - data_min) * (np.ones(n_spatial_dims) - np.zeros(n_spatial_dims)) / (data_max - data_min)
    
    return X_curr_view_warped


def sim_multiple_slices_allsims_tissue_forbatch(adata_st,types_of_spots=['linear','polar','gaussian'],
                        num_simulations=3,
                        spa_celltype_key='Types',N=50, # save_spot_in_slices='spots_simulated.pickle',
                        save_sim_slices='warped_spots_simulated.pickle',wts=None,
                        numcells_to_swap=2, num_tissueslices = 10,spot_base_key='spot_number',
                        linear_slope_variance=0.9,linear_intercept_variance=0.9,
                        polar_slope_variance=0.9,
                        gauss_kernel_variance=1.0,gauss_kernel_lengthscale=1.0,
                                            gauss_mean_slope=1.0,
                                            gauss_mean_intercept=0.0, # hard code in function
                        # linear_warp_slope_seed=1,linear_warp_int_seed=1,   # hard code in function
                        # polar_warp_seed=1,
                        spatial_base_key = 'spatial',
                        polar_noise_seed=1, # hard code in function
                        linear_noise_seed=1,
                        gauss_noise_seed=1,
                        gauss_seed=1, 
                        noise_mean=np.zeros(2),
                        noise_var_coeff= 0.0005,
                        swap_seed = 1
                        ):
    
    # note that if we decide to swap too many spots, it won't make a difference as the maximum number 
    # will be swapped in this version of the code 

    adata_orig = {}
    adata_spot_swap = {}
    types_of_sims = []

    # parameters for seeds 
    seed_par = [23]
    for sim_index in range(int(num_simulations)):
        for entry in types_of_spots:
            types_of_sims_curr = entry + '_' + str(sim_index)
            adata_orig[types_of_sims_curr] = {}
            # adata_spot[types_of_sims_curr] = {}
            adata_spot_swap[types_of_sims_curr] = {}
            types_of_sims.append(types_of_sims_curr)

            for entry2 in range(num_tissueslices+1): # this dict stores the different warps, add 1 since the first entry is the original slice
                adata_orig[types_of_sims_curr][entry2] = {}
                adata_spot_swap[types_of_sims_curr][entry2] = {}
                
                seed_par.append(seed_par[-1]+1) # some random number 

            for slice_idx in adata_st.keys():
                adata_orig[types_of_sims_curr][0][slice_idx], adata_spot_swap[types_of_sims_curr][0][slice_idx] = make_spot(adata_st[slice_idx],
                                                                                                                            N=N,
                                                                                                                            wts=wts, 
                                                                                                                            spa_celltype_key=spa_celltype_key,
                                                                                                                            spatial_key=spatial_base_key,
                                                                                                                            spot_key=spot_base_key)

    seed_par=seed_par[0:-1]
    # print(len(seed_par))

    # print(types_of_sims)
    
    
    for slice_idx in adata_st.keys():
        cnt = 0    

        for entry0 in range(num_tissueslices):
            
            # with every successive slice, I move the original to swapped and delete swapped so we dont need this 
            # if entry0 != 0:
            #     spatial_key = spatial_base_key + '_warped' # for the first one, it isnt warped, after it is 
            #     spot_key = spot_base_key + '_swapped'
            # else:
            #     spatial_key = spatial_base_key #+ '_warped' # for the first one, it isnt warped, after it is 
            #     spot_key = spot_base_key #+ '_swapped'

            for entry in types_of_sims:
                # print(entry, slice_idx, entry0)
                if 'linear' in entry: # linear warp
                    # print(adata_spot_swap[entry][entry0][slice_idx])
                    warped_spatial = make_warp_linear(
                        adata_spot_swap[entry][entry0][slice_idx], # 
                        # warp_type='linear',
                        spatial_key = spatial_base_key, # 
                        linear_slope_variance=linear_slope_variance, # 
                        linear_intercept_variance=linear_intercept_variance, # 
                        linear_warp_slope_seed=seed_par[cnt],  # vary  # 
                        linear_warp_int_seed=seed_par[cnt], # set same as linear_warp_slope_seed # 
                        linear_noise_seed=linear_noise_seed, # 
                        noise_mean=noise_mean, # 
                        noise_var_coeff=noise_var_coeff # 
                        )
                    #print(warped_spatial)
                elif 'polar' in entry: # polar warp
                    warped_spatial = make_warp_polar(
                        adata_spot_swap[entry][entry0][slice_idx],
                        # warp_type='polar',
                        spatial_key = spatial_base_key, 
                        polar_slope_variance=polar_slope_variance,
                        polar_warp_seed=seed_par[cnt], # vary but keep same as linear
                        polar_noise_seed=polar_noise_seed,
                        noise_mean=noise_mean,
                        noise_var_coeff=noise_var_coeff
                        )
                    # print(warped_spatial.head())
                elif 'gaussian' in entry:
                    warped_spatial = make_warp_gaussian(
                        adata_spot_swap[entry][entry0][slice_idx],
                        #warp_type='gaussian',
                        spatial_key = spatial_base_key, 
                        gauss_kernel_variance=gauss_kernel_variance,
                        gauss_kernel_lengthscale=gauss_kernel_lengthscale,
                        gauss_mean_slope=gauss_mean_slope,
                        gauss_mean_intercept=gauss_mean_intercept,
                        gauss_seed=seed_par[cnt], # vary but keep same as linear
                        noise_mean= noise_mean,
                        noise_var_coeff=noise_var_coeff, 
                        gauss_noise_seed=gauss_noise_seed)
                    #print(warped_spatial.head())
                else:
                    print("shouldnt be here; invalid simulation")

                # honestly this line is only for debugging purposes: 
                # the next slice should have spatial_key = spatial_key + '_warped'
                adata_spot_swap[entry][entry0][slice_idx].obsm[spatial_base_key + '_warped'] = warped_spatial

                # print(adata_orig[entry][entry0][slice_idx])
                # print(adata_spot_swap[entry][entry0][slice_idx])
                # print(numcells_to_swap)
                # print(spot_key)
                # print(spa_celltype_key)

                adata_orig[entry][entry0+1][slice_idx], adata_spot_swap[entry][entry0+1][slice_idx] = cell_swap_for_allsims_tissue(adata_orig[entry][entry0][slice_idx], 
                                                                        adata_spot_swap[entry][entry0][slice_idx],
                                                                        numcells_to_swap=numcells_to_swap, 
                                                                        spot_key = spot_base_key, # 'spot_number' for the 1st, spot_key_warped after
                                                                        spa_celltype_key=spa_celltype_key,
                                                                        swap_seed=swap_seed,
                                                                        wts=wts) 

                # doing this so the next slice in the sequence will be correctly updated 
                adata_spot_swap[entry][entry0+1][slice_idx].obsm[spatial_base_key] = adata_spot_swap[entry][entry0+1][slice_idx].obsm[spatial_base_key + '_warped'].copy()
                del adata_spot_swap[entry][entry0+1][slice_idx].obsm[spatial_base_key + '_warped']
                


                cnt = cnt + 1

    
    # for entry2 in types_of_sims:
    #     for entry in range(num_tissueslices+1):
    #         if (len(adata_spot_swap[entry2][entry]) == 0):
    #               del adata_spot_swap[entry2][entry]
    #     if (len(adata_spot_swap[entry2]) == 0):
    #         del adata_spot_swap[entry2]

    #if (len(adata_spot_swap.keys()) > 0):
    with open(save_sim_slices, 'wb') as handle:
        pickle.dump(adata_spot_swap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return adata_orig,  adata_spot_swap # adata_spot,

