from .importing_modules import *
from .processing_for_model import *
from .post_processing import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################   use other tools to align adjacent slices #############
def solve_alignment(adata_sp, reference=None, policy='star'):
    # star aligns to reference
    # sequential aligns to the next 

    ap = AlignmentProblem(adata=adata_sp)
    ap = ap.prepare(batch_key="batch", policy=policy,reference=reference)
    ap = ap.solve()
    
    return ap
############################################################################


# optional to check convergence


def compute_cost_function_ms(V, res1, alpha, tissue_expr, # X1 is a list of slices 
                          # X2, 
                          # curr_slice_idx, 
                          Bf, tissue_kernel, # P, 
                            # ct_identity, 
                            lambda_,
                           # eta1_,  
                           adj_wt_,ct_props_prior=None):
    """
    """
    
    res1_softmax = torch.nn.functional.softmax(res1,dim=0)
    # print("res1_softmax")
    # print(res1_softmax.shape)
    # print("cells x ctypes x spots")
    # why are we applying softmax to columns of res 1 ? -> ct_identity is column normalized i.e. each col sums to 1 so it's like "averaging"
    # hence since we're keeping res1 close to ct_identity, we need to column normalize this too 
    
    ####################### tensorize bv_prod_X1_main #################################
    # Bf = genes x cells; res1_softmax = cells x ctypes x spots; V = ctypes x spots 
    # print("Bf")
    # print(Bf.shape)
    # print(" genes x cells")

    # print("V")
    # print(V.shape)
    # print(" ctypes x spots")

    tmp_prod1 = torch.matmul(Bf, res1_softmax.view(res1_softmax.shape[0], -1))  # torch.einsum('gc,cts->gts', Bf, res1_softmax)
    # print("tmp_prod1")
    # print(tmp_prod1.shape)
    # print("genes, ctypes x spots")

    tmp_prod1 = tmp_prod1.view(Bf.shape[0], res1_softmax.shape[1], res1_softmax.shape[2]) # genes x   ctypes x spots 
    # print("tmp_prod1")
    # print(tmp_prod1.shape)
    # print("genes x ctypes x spots")
    # # bv_prod_X1_main = torch.einsum('gts,ts->gs', tmp_prod1, torch.nn.functional.softmax(V, dim=0)) 
    # tmp_prod1 is genes x ctypes x spots; V is ctypes x spots 
    # print(tmp_prod1.shape, torch.nn.functional.softmax(V, dim=0).unsqueeze(0).shape)
    # V_mod = torch.nn.functional.softmax(V, dim=0) * ct_props_prior
    # V_mod = V_mod / V_mod.sum(dim=0, keepdim=True)
    bv_prod_X1_main = (tmp_prod1*torch.nn.functional.softmax(V, dim=0).unsqueeze(0)).sum(dim=1) # genes x ctypes x spots * 1 x ctypes x spots = genes x ctypes  x spots -> genes x spots
    # print("bv_prod_X1_main")
    # print(bv_prod_X1_main.shape)
    # print("genes x spots")

    # exapnd for use in subtraction
    bv_prod_X1_main = bv_prod_X1_main.unsqueeze(0).unsqueeze(2)
    bv_prod_X1_main = bv_prod_X1_main.expand(tissue_expr.shape[0], -1, tissue_expr.shape[2], -1)
    kernel_tmp = tissue_kernel.unsqueeze(1).expand(-1, tissue_expr.shape[1], -1, -1)
    bv_prod_X1_main = bv_prod_X1_main * ((kernel_tmp > 0.0).float()) # multiply by binarized kernel
    # print("bv_prod_X1_main")
    # print(bv_prod_X1_main.shape)
    # print("slices x genes x numadpspots x numspots ")
    # # bv_prod_X1_main: this is slices x genes x numadpspots x numspots 
    ######################### tensorize bv_prod_X1_main ends ###########################


    # adj_wt_ is slices x 1 
    # alpha is 1 x 1 x 1 x numspots 
    # term 1 should be of length "spots"
    # previously first one was "sum" instead of "mean"
    term1 = 0.5* (adj_wt_ * (tissue_kernel * ((tissue_expr - torch.clamp(alpha, min = 1e-4) * bv_prod_X1_main).pow(2).mean(dim=1))).sum(dim=1)).sum(dim=0)
    # print()
    # print("term1")
    # print(term1.shape)
    # print("numspots")
    # V = ctypes x spots 
    # V should be of length spots 
    l2_term = 0.5 *  lambda_ * (V.pow(2).mean(dim=0)) # previously sum
    # print("l2_term")
    # print(l2_term.shape)
    # print("numspots")

    # res1_term -> 0
    # res1_softmax = cells x ctypes x spots
    # ct_identity = cells x ctypes x spots
    # res1_term = 0.5 * eta1_ * ((res1_softmax-ct_identity).pow(2).mean(dim=0).mean(dim=0)) 
    # print("res1_term")
    # print(res1_term.shape)
    # print("numspots")

    # Total cost
    cost = term1 +  l2_term # + res1_term 
    # print("cost")
    # print(cost.sum())
    # print("1 value")
    # print(f"{term1.mean().item():.3f}") 
    # print(f"{l2_term.mean().item():.3f}")
    # print(f"{res1_term.mean().item():.3f}")
    # print(term1, l2_term)
    return cost.mean() # , term1_net,  term1_list, l2_term, res1_term # term2,  , ,  res2_term



def run_adam_softmax_optimization_ms(Bf,  X1_list, # curr_slice_idx, # X2,  
                                  w_list, #P, 
                                  ct_identity, adj_wt_, res1_init, # ,alpha_init, # res2_init,
                                  V_init,
                       lambda_, eta1_, 
                        max_iter_adam=500,#  Nimp=3,
                        par_lr_adam=1e-2, batch_sz=500,
                         cell_mode='sc',ct_props_prior=None ):

    numspots = w_list[0].shape[1]
    # print(numspots)

    numgenes = X1_list[0].shape[0]

    output_slices = []
    kernel_slices = []
    # kernel_binarized = []

    ######################## tensorize #############################
    for slice_num in range(len(X1_list)):
        max_neigh = (w_list[slice_num] != 0).sum(axis=0).max().item() # max size for padding
        # print(max_neigh)
        # define output tensors 
        # numspots is number of spots in current slice
        output_tensor = torch.full((numgenes, max_neigh, numspots), 0.0, dtype=X1_list[slice_num].dtype, device=X1_list[slice_num].device)
        kernel_tensor = torch.full((max_neigh, numspots), 0.0, dtype=w_list[slice_num].dtype, device=w_list[slice_num].device)
        # kernel_binarized = torch.full((max_neigh, numspots), 0.0, dtype=w_list[slice_num].dtype, device=w_list[slice_num].device)
        # print(output_tensor.shape)
        # print(kernel_tensor.shape)
        for col_idx in range(numspots):
            batch_id = (w_list[slice_num][:,col_idx] != 0).squeeze() # need the : so col_idx denotes columns ( tested this )
            extracted_kernel = w_list[slice_num][batch_id,col_idx]
            # extracted_kernelbin =  (extracted_kernel > 0.0).float() * 1
            extracted_values = X1_list[slice_num][:, batch_id]
            # Store extracted values in the output tensor
            output_tensor[:, :extracted_values.shape[1], col_idx] = extracted_values
            kernel_tensor[:extracted_kernel.shape[0], col_idx] = extracted_kernel

            # print(output_tensor.shape, kernel_tensor.shape) 
            # kernel_binarized[:extracted_kernelbin.shape[0], col_idx] = extracted_kernelbin

        output_slices.append(output_tensor)
        kernel_slices.append(kernel_tensor)
        # kernel_binarized.append(kernel_binarized)

    # now convert to padded 4D slice tensor
    max_adjspots = max(mat.shape[1] for mat in output_slices) 
    tissue_expr = torch.full((len(output_slices), numgenes, max_adjspots, numspots), fill_value=0.0, dtype=output_slices[0].dtype,device=output_slices[0].device)
    tissue_kernel = torch.full((len(kernel_slices), max_adjspots, numspots), fill_value=0.0, dtype=kernel_slices[0].dtype,device=kernel_slices[0].device)
    # tissue_kernel_bin = 

    for i, mat in enumerate(output_slices):
        ngenes_curr, nadj_spots, ncurrspots = mat.shape
        tissue_expr[i, :, :nadj_spots, :] = mat
        tissue_kernel[i, :nadj_spots, :] = kernel_slices[i]

    # for memory issues - didnt help
    del output_slices, kernel_slices
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    ######################## tensorization ends ####################

    # process spots in batches 
    V_batch_list = []
    alpha_batch_list = []
    res1_batch_list = []

    for start in range(0, numspots, batch_sz):
        end = min(start + batch_sz, numspots)
        bs  = end - start

        V_batch = V_init[:, start:end].clone().requires_grad_(True).to(device) # shape is ctypes x spots 
        # res1_batch = res1_init[:, :, start:end].clone().requires_grad_(True).to(device)  # shape should be cells x celltypes x spots  
        # tissue_expr.shape[0]
        alpha_batch = torch.ones(1, 1, 1, bs, requires_grad=True, device=device) # shape is 1 x 1 x 1 x spots 
        expr_batch  = tissue_expr[:, :, :, start:end]
        kern_batch  = tissue_kernel[:, :,   start:end]
        ct_props_prior_batch = ct_props_prior[:,start:end]
        # scaler = torch.cuda.amp.GradScaler() # added for memory

        # I want each spot to function as a separate sample 
        if cell_mode == 'sc':
            res1_batch = res1_init[:, :, start:end].clone().requires_grad_(True).to(device)  # shape should be cells x celltypes x spots  
            optimizer = torch.optim.Adam([V_batch, res1_batch, alpha_batch], lr=par_lr_adam)
        elif cell_mode == 'cluster':
            res1_batch = res1_init[:, :, start:end].clone().to(device)  # shape should be cells x celltypes x spots  
            optimizer = torch.optim.Adam([V_batch,  alpha_batch], lr=par_lr_adam)
        else:
            print(cell_mode + " cell mode not found")
            sys.exit()
        prev_cost = torch.tensor(float("inf"), device=device)

        for iteration in range(max_iter_adam):
            
            optimizer.zero_grad()
            
            ############### cost function call ##################
            cost  = compute_cost_function_ms(V_batch, res1_batch, alpha_batch, 
                                             expr_batch,  Bf, kern_batch, # ct_identity, 
                                             lambda_,# eta1_,
                                            adj_wt_=adj_wt_,ct_props_prior=ct_props_prior_batch)
            # print(cost, iteration)
            # with checkpoint ( for memory issues at backward - didnt help ) 
            # with torch.cuda.amp.autocast():
            #     cost = checkpoint(
            #         compute_cost_function_ms,
            #         V_batch, res1_batch, alpha_batch,
            #         expr_batch, Bf, kern_batch,
            #         # ct_identity, 
            #         lambda_, 
            #         # eta1_, 
            #         adj_wt_
            #     )
            # scaler.scale(cost).backward()
            # scaler.step(optimizer)
            # scaler.update()
            #################################################                                                                
            cost.backward()
            optimizer.step()

            # Stopping criterion
            if ((iteration >= 2) and (torch.abs(prev_cost - cost) < 1e-4)):
                break
            prev_cost = cost.detach().clone()

        # V_mod = torch.nn.functional.softmax(V_batch, dim=0) * ct_props_prior_batch    
        # V_mod = V_mod / V_mod.sum(dim=0, keepdim=True)
        V_batch_list.append(torch.nn.functional.softmax(V_batch, dim=0).detach())
        alpha_batch_list.extend(alpha_batch.squeeze().detach().cpu().numpy().tolist())
        res1_batch_list.append(torch.nn.functional.softmax(res1_batch,dim=0).detach())
    
    V_full = torch.cat(V_batch_list, dim=1)    
    # alpha_full = torch.cat(alpha_batch_list, dim=1)
    res1_full = torch.cat(res1_batch_list, dim=2)
    del res1_batch_list
 
    # apply softmax 
    # V = torch.nn.functional.softmax(V_full, dim=0) # without dim=0, the outputs was all 1s
    # res1_full = torch.nn.functional.softmax(res1_full,dim=0)
    # cell_wts = res1 @ V
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return V_full.detach().cpu().numpy(), res1_full.detach().cpu().numpy(), alpha_batch_list # alpha_full.detach().cpu().numpy() # , cell_wts.detach().cpu().numpy() #, res1.detach().cpu().numpy(), alpha.detach().cpu().numpy(), all_debug_vars 



def spadecoder_slice_wrapper_ms(adata_ip, # this is a list 
                                curr_slice_idx, 
                                Bsc, ct_identity,
                                spa_key1='spatial',
                                min_wt=0.0001,
                                renorm=True,
                                bandwidth=0.01,
                                recompute=True,par_lambda=0.001, 
                                par_eta1=10.0,
                                max_iter_adam=500,
                                n_spatial_neigh=15,nn_only=True,
                                weight_spatial=1.0,
                                adj_wt=None,
                                ct_props=None,
                                par_lr_adam=0.01,
                                adata_bulk_init=None, 
                                kernel3d_bw_slices=100, # keep_top_align=1,# ap=None,
                                # res1_init = None, alpha_init = None,
                                gt_align=True,
                                Palign_ip = None, # this is alignment distance NOT alignment probability 
                                kernel_ip = None,
                                adj_wts = None,
                                kernel_wt_list = None,
                                batch_sz=500,
                                cell_mode='sc'):

    # gt_align: If true, ground truth alignment is used. If False, alignment is computed using moscot
    # ensure same genes in ST, scRNA
    # Bsc: gene X cell matrix from reference 
    # ct_identity: cell X cell-type binary matrix 
    
    ##### genes, cells same  #####################
    genes_to_use = list(set(adata_ip[0].var.index).intersection(set(Bsc.index)))
    for entry in range(1,len(adata_ip)):
        genes_to_use = set(adata_ip[entry].var.index).intersection(genes_to_use)
    genes_to_use = list(genes_to_use)
    genes_to_use.sort()

    for entry in range(len(adata_ip)):
        adata_ip[entry] = adata_ip[entry][:,genes_to_use].copy()
    
    Bsc = Bsc.loc[genes_to_use,:].copy()
   
    # add zeros if cell-types not present
    # cell-types restricted by what's in reference
    all_ctypes = list(ct_identity.columns) # .union(set(adata_st1.obs.columns).union(set(adata_st2.obs.columns))))

    for entry0 in range(len(adata_ip)):
        if  set(adata_ip[entry0].obs.columns) != set(all_ctypes):
            # add extra columns with 0s
            for entry in all_ctypes:
                if entry not in set(adata_ip[entry0].obs.columns):
                    obs =  sc.get.obs_df(adata_ip[entry0],keys=list(adata_ip[entry0].obs.columns))
                    obs[entry] = 0.0
                    adata_ip[entry0].obs = obs
    #######################################################################################

    num_curr_slices = len(adata_ip)

    ###############################  calculate 3D weights ###########################################
    if adj_wts is None:
        if num_curr_slices > 1: # many slices 
            two_sig = int(kernel3d_bw_slices/2) # make sure the maximum number of important slices are covered in 2-sigma
            sigma = two_sig/2.0
            
            
            adj_wts = []
            for slice_idx in range(num_curr_slices):
                dist_from_curr = abs(slice_idx-curr_slice_idx)
                slice_wt = gaussian_kernel_for3d(dist_from_curr, sigma)
                #print(sigma, dist_from_curr, slice_idx,curr_slice_idx,slice_wt)
                adj_wts.append(slice_wt)
            adj_wts = np.array(adj_wts)/np.sum(adj_wts) # previously max
            # renormalize so max is at current slice
        else:
            adj_wts = np.array([1.0])
    
    # print(adj_wts)
    # print(kernel_wt_list)

    adj_wts =  torch.tensor(adj_wts, dtype=torch.float32,requires_grad=False, device=device).unsqueeze(1)
    # print(adj_wts)
    ############### end 3D weight calculation #######################


    ####################################  2D weights #################################################
    if kernel_wt_list is None:
        kernel_wt_list = []
        for entry in range(num_curr_slices):
            if entry == curr_slice_idx:
                if kernel_ip is None:
                    kernel_wt1 = get_gauss_kernel_wt(adata_ip[curr_slice_idx],spa_key1,min_wt=min_wt,bandwidth=bandwidth,
                                                                recompute=recompute,n_spatial_neigh=n_spatial_neigh,
                                                                nn_only=nn_only,weight_spatial=weight_spatial) 
                    
                    # for abaltion only 
                    # kernel_wt1 = np.zeros((adata_ip[curr_slice_idx].shape[0],adata_ip[curr_slice_idx].shape[0]))
                    # np.fill_diagonal(kernel_wt1, weight_spatial) 

                    #### DEBUGGING ONLY, REMOVE
                    # set 5 weights to 1 
                    # renormalize 
                    # kernel_wt1 = (kernel_wt1 > 0).astype(int)
                    # print(kernel_wt1.sum(axis=0).max(),kernel_wt1.sum(axis=0).min())
                    # kernel_wt1 = (kernel_wt1/kernel_wt1.sum(axis=0))
                else:
                    kernel_wt1 = kernel_ip.copy()
                # print((kernel_wt1 > 0).astype(int).sum(axis=0)) # check number of neighbors 
            else:
                if gt_align: # GT alignment, fill with 1's on diagonal
                    
                    ## TEST!!! - nbd of neighboring slice
                    # kernel_wt1 = get_gauss_kernel_wt(adata_ip[entry],spa_key1,min_wt=min_wt,bandwidth=bandwidth,
                    #                                         recompute=recompute,n_spatial_neigh=n_spatial_neigh,
                    #                                         nn_only=nn_only,weight_spatial=weight_spatial) 
                    ## ORIGINAL!!
                    kernel_wt1 = np.zeros((adata_ip[entry].shape[0],adata_ip[curr_slice_idx].shape[0]))
                    
                    
                    # print(adata_ip[entry].shape[0],adata_ip[curr_slice_idx].shape[0])
                    # np.fill_diagonal(kernel_wt1,0.75)
                    np.fill_diagonal(kernel_wt1,1.0)
                else:
                    # run PASTE To align everything with "curr_slice_idx"
                    Palign_dis = Palign_ip[(str(entry),str(curr_slice_idx))] # this is of dim aligned slice x currslice 
                    
                    
                    # added for test
                    kernel_wt1 = get_gauss_kernel_wt_neighslice(Palign_dis, min_wt=min_wt,bandwidth=bandwidth,
                                                                recompute=recompute,n_spatial_neigh=n_spatial_neigh,
                                                                nn_only=nn_only,weight_spatial=weight_spatial)
                    

                    
                    ## MODIFIED 
                    # kernel_wt1 = keep_top_n_per_column(Palign, keep_top=keep_top)
                    # print(kernel_wt1.shape)

            # print(entry, kernel_wt1)
            kernel_wt_list.append(torch.tensor(kernel_wt1,dtype=torch.float32,requires_grad=False).to(device))
    # print(kernel_wt_list)
    ###################################### 2D weights end ######################################

    #### make into tensors #######################
    numspots = adata_ip[curr_slice_idx].shape[0]

    Bscrna = torch.tensor(np.array(Bsc), dtype=torch.float32,requires_grad=False).to(device)
    ct_identity = torch.tensor(np.array(ct_identity), dtype=torch.float32,requires_grad=False).to(device) # cells x ctypes 
    ct_identity = ct_identity.unsqueeze(2).expand(-1, -1, numspots) # cells x ctypes x numspots 
    X1_list = [torch.tensor(entry.X.T, dtype=torch.float32,requires_grad=False).to(device) for entry in adata_ip]
    ##############################################################
    
    n_celltypes = len(all_ctypes)
    n_spots_st1 = X1_list[curr_slice_idx].shape[1]
    
    # ref cells/clusters x query spots/cells matrix
    # cell_wts_df = pd.DataFrame(np.zeros((Bscrna.shape[1],n_spots_st1)),index=range(Bscrna.shape[1]),columns=range(n_spots_st1))

    ############### get initializtion of cell-type props - V_init #######################
    if adata_bulk_init is  None:
        if ct_props is None:
                V_init = torch.rand((n_celltypes, n_spots_st1), dtype=torch.float32, device=device)
                V_init /= V_init.sum(dim=0, keepdim=True)
                # print("ctprops  None, ")
                # print(V_init.shape)
        else:
                V_init = torch.tensor(ct_props[all_ctypes].values, dtype=torch.float32, device=device).reshape(-1,1).repeat(1,n_spots_st1) #.repeat(1, n_spots_st1)
                # print("ctprops not None, ")
                # print( V_init.shape)
    else:
        V_init = torch.tensor(adata_bulk_init.loc[all_ctypes,:].values,dtype=torch.float32,requires_grad=False,device=device)
        # print("adata_bulk_init not None, ")
        # print(V_init.shape)
    # ctypes x spots 
    ct_props_prior = torch.tensor(ct_props[all_ctypes].values, dtype=torch.float32, device=device).reshape(-1,1).repeat(1,n_spots_st1)
    ################ V_initialization ends ########################
    
    
    ############## initialize residual #####################
    # initlaize same as ct-identity so close to softmax is close to ctype 
    # res1_init = torch.log(ct_identity) # this gets to infiniy but with the softmax, it's fine
    res1_init = torch.log(ct_identity)   # Shape: (cells , celltypes x numspots)
    #########################################################
    

    deconv_op, res1_full, alpha_full = run_adam_softmax_optimization_ms(Bscrna, X1_list, # curr_slice_idx,
                                                 kernel_wt_list, ct_identity, adj_wts,
                                                 res1_init=res1_init,
                                                 V_init=V_init,
                                                 lambda_=par_lambda,
                                                 eta1_=par_eta1,
                                                 max_iter_adam=max_iter_adam,
                                                 par_lr_adam=par_lr_adam, batch_sz=batch_sz,cell_mode=cell_mode,
                                                 ct_props_prior=ct_props_prior)
    
    
    deconv_op = pd.DataFrame(deconv_op, index=all_ctypes, columns=adata_ip[curr_slice_idx].obs.index)    
    return deconv_op, res1_full, alpha_full # , cell_wts # , res1_1, alpha, debug_vars1



def keep_top_n_per_column(matrix, keep_top):
    # Create a copy to avoid modifying the original matrix
    result = np.zeros_like(matrix)
    # Process each column individually
    for col in range(matrix.shape[1]):
        col_values = matrix[:, col]
        # If n is less than the number of rows, find the top n indices
        if keep_top < len(col_values):
            # argpartition puts the top n values in the last n positions (unsorted)
            top_n_indices = np.argpartition(col_values, -keep_top)[-keep_top:]
        else:
            # If n exceeds or equals the number of rows, keep all values
            top_n_indices = np.arange(len(col_values))
        # Retain only the top n values in the result
        result[top_n_indices, col] = matrix[top_n_indices, col]

        col_sum = result[:, col].sum()
        if col_sum != 0:
            result[:, col] /= col_sum
    return result