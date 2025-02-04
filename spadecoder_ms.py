from .importing_modules import *
from .processing_for_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################   use other tools to align adjacent slices #############
def solve_alignment(adata_sp):
    ap = AlignmentProblem(adata=adata_sp)
    ap = ap.prepare(batch_key="batch", policy="sequential")
    ap = ap.solve()
    
    return ap
############################################################################


# optional to check convergence


def compute_cost_function_ms(V, res1, alpha, X1_list, # X1 is a list of slices 
                          # X2, 
                          curr_slice_idx, Bf, w_list, # P, 
                            ct_identity, lambda_,
                           eta1_,  adj_wt_=None):
    """
    Compute the cost function.
    
    :param B: Gene expression matrix (G x K)
    :param X1: Gene-by-spot matrix for sample S1 (G x N1)
    :param X2: Gene-by-spot matrix for sample S2 (G x N2)
    :param V1: Cell-type proportion matrix for slice 1 (K,)
    
    :param w: Weights for X1 term (N1,)
    :param P: Weights for X2 term (N2,)
    :param lambda_: Regularization parameter
    :param alpha_: Elastic net tuning parameter (Between L1 and L2)
    :param self_wt_: tuning the weight of current slice term
    :param adj_wt_: tuning the weight of adjacent slice term
    :return: Computed cost
    """
    
    res1_softmax = torch.nn.functional.softmax(res1,dim=0)
    # why are we applying softmax to columns of res 1 ? -> ct_identity is column normalized i.e. each col sums to 1 so it's like "averaging"
    # hence since we're keeping res1 close to ct_identity, we need to column normalize this too 
    bv_prod_X1_main =  Bf @  res1_softmax @  torch.nn.functional.softmax(V, dim=0) #bv_prod.expand(-1, X1.shape[1])
    # bv_prod_X1 = bv_prod_X1.unsqueeze(2).expand(-1, -1, X1_tensor.shape[2])

    if adj_wt_ is None:
        adj_wt_ = torch.ones(len(X1_list),dtype=torch.float32,requires_grad=False, device=device)

    # term1 = adj_wt_ * 0.5 * (w_tensor * (X1_tensor - torch.clamp(alpha, min=1e-4) * bv_prod_X1).pow(2)).sum(dim=1).mean()
    # term1_list = []
    # term1_net = 
    # for slice 0 
    batch_id = (w_list[0] != 0).squeeze()
    X1_batch = X1_list[0][:, batch_id]
    # G x N 
    bv_prod_X1 = bv_prod_X1_main.expand(-1, X1_batch.shape[1])
    # Compute the first term: sum of weighted squared L2 norms for X1
    term1_net = adj_wt_[0] *  0.5 * (w_list[0][batch_id] * (X1_batch -  torch.clamp(alpha, min = 1e-4) * bv_prod_X1).pow(2).sum(0)).mean()
    # term1_list = [term1_net]
        
    for slice_num in range(1, len(X1_list)):
        batch_id = (w_list[slice_num] != 0).squeeze()
        X1_batch = X1_list[slice_num][:, batch_id]
        # G x N 
        bv_prod_X1 = bv_prod_X1_main.expand(-1, X1_batch.shape[1])
        # Compute the first term: sum of weighted squared L2 norms for X1
        term1 = adj_wt_[slice_num] *  0.5 * (w_list[slice_num][batch_id] * (X1_batch -  torch.clamp(alpha, min = 1e-4) * bv_prod_X1).pow(2).sum(0)).mean()
        # term1_list.append(term1)
        term1_net += term1

    # Compute the third term: L2 regularization term
    l2_term = 0.5 *  lambda_ * (V.pow(2).sum())

    # res1_term -> 0
    res1_term = 0.5 * eta1_ * ((res1_softmax-ct_identity).pow(2).mean()) 

    # Total cost
    cost = term1_net +  l2_term + res1_term 

    return cost # , term1_net,  term1_list, l2_term, res1_term # term2,  , ,  res2_term


def run_adam_softmax_optimization_ms(Bf,  X1_list, curr_slice_idx, # X2,  
                                  w_list, #P, 
                                  ct_identity, res1_init, # res2_init,
                                  V_init,
                       lambda_, eta1_, 
                        # self_wt_=1.0,
                        adj_wt_=None, 
                        max_iter_adam=500, 
                        par_lr_adam=1e-2 ):
    """
    Adam optimization without ADMM.
    
    :param B: Gene expression matrix (G x K)
    :param X1: Gene-by-spot matrix for sample S1 (G x N1)
    :param X2: Gene-by-spot matrix for sample S2 (G x N2)
    :param V_init: Initial cell-type proportion matrix (K,)
    :param w: Weights for X1 term (N1,)
    :param P: Weights for X2 term (N2,)
    :param lambda_: Regularization parameter
    :param alpha_: Elastic net tuning parameter (Between L1 and L2)
    :param max_iter: Maximum number of iterations
    :param printiter: Print interval for logging
    :param par_lr: Learning rate for Adam optimizer
    :param dist_measure: Distance measure ('L2' or 'cosine')
    :return: Optimized V and bias
    """

    V = V_init.clone().requires_grad_(True).to(device) 
    res1 = res1_init.clone().requires_grad_(True).to(device) 
    alpha = torch.tensor([1.0], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([V, res1, alpha], lr=par_lr_adam)
    prev_cost = torch.tensor(float("inf"), device=device)
    
    # for debugging 
    # cost_list = []
    # term1_list_net = []
    # term1_list_adj = []
    # l2_term_list = []
    # res1_term_list = []

    for iteration in range(max_iter_adam):
        optimizer.zero_grad()
        # cost,term1_net, term1_adjs, l2_term, res1_term   = compute_cost_function_ms(V, res1,alpha, X1_list, curr_slice_idx,
        cost  = compute_cost_function_ms(V, res1,alpha, X1_list, curr_slice_idx,
                                                                        Bf, w_list, ct_identity, lambda_,eta1_)
                                                                        # self_wt_=self_wt_)
        
        # for debugging
        # cost_list.append(cost)
        # term1_list_net.append(term1_net)
        # term1_list_adj.append(term1_adjs)
        # # term2_list.append(term2)
        # l2_term_list.append(l2_term)
        # res1_term_list.append(res1_term)
        # all_debug_vars = (cost_list,term1_list_net,term1_list_adj, l2_term_list ,res1_term_list)
        # # debugging ends 

        cost.backward()
        optimizer.step()

        # Stopping criterion
        # cost_float = cost.item()
        if torch.abs(prev_cost - cost) < 1e-4:
            break
        prev_cost = cost.clone()
 
    # apply softmax 
    V = torch.nn.functional.softmax(V, dim=0) # without dim=0, the outputs was all 1s
    res1 = torch.nn.functional.softmax(res1,dim=0)
    cell_wts = res1 @ V
    
    return V.detach().cpu().numpy(), cell_wts.detach().cpu().numpy() #, res1.detach().cpu().numpy(), alpha.detach().cpu().numpy(), all_debug_vars 


def get_adam_softmax_solution_perslice_ms(kernel_wt_list,curr_slice_idx, Bscrna, ct_identity,
                                    all_ctypes, expr_st1_list, 
                                    par_lambda=1.0, 
                                    par_eta1=5.0, 
                                    max_iter_adam=500, 
                      adj_wt_=None,
                      ct_props=None,
                      par_lr_adam=1e-2,
                      adata_bulk_init = None):
    
    n_celltypes = len(all_ctypes)
    n_spots_st1 = expr_st1_list[curr_slice_idx].shape[0]
    deconv_df = pd.DataFrame(np.zeros((n_celltypes,n_spots_st1)),index=all_ctypes,columns=range(n_spots_st1))

    # ref cells/clusters x query spots/cells matrix
    cell_wts_df = pd.DataFrame(np.zeros((Bscrna.shape[1],n_spots_st1)),index=range(Bscrna.shape[1]),columns=range(n_spots_st1))

    # V_init
    if adata_bulk_init is  None:
        if ct_props is None:
                V_init = np.random.rand(n_celltypes)
                V_init = V_init/V_init.sum()
        else:
                V_init = ct_props[all_ctypes].values # make sure same order as Bscrna

        # V_init = np.array(V_init).reshape(-1,1)
        V_init = torch.tensor(V_init,  dtype=torch.float32,requires_grad=False).reshape(-1,1).to(device)
    else:
        adata_bulk_init = adata_bulk_init.loc[all_ctypes,:].copy()

    X1_list = [torch.tensor(expr_st1_list[entry].T, dtype=torch.float32,requires_grad=False).to(device) for entry in range(len(expr_st1_list))]  # Ensure X1 is a PyTorch tensor
    
    # print(kernel_wt_list)
    kernel_wt_list = [torch.tensor(kernel_wt_list[entry], dtype=torch.float32,requires_grad=False).to(device) for entry in range(len(kernel_wt_list))]  # Ensure w is a PyTorch tensor
  
    # initlaize same as ct-identity so close to softmax is close to ctype 
    res1_init = torch.log(ct_identity) # this gets to infiniy but with the softmax, it's fine
    
    # debugging only 
    debug_vars_list_spots = []

    for spt_num in range(n_spots_st1):  
        if adata_bulk_init is not None:
            #print(adata_bulk_init)
            # note that spatial location has spt_num as an int but deconv has spt_num as a str - fixed by making a copy in evaluations
            # since for deconv the names of cells were getting updated in evaluations 
            # V_init = np.array(adata_bulk_init.loc[all_ctypes,spt_num]).reshape(-1,1)
            V_init = torch.tensor(adata_bulk_init[spt_num].values,dtype=torch.float32,requires_grad=False).reshape(-1,1).to(device)
        
        kernel_wt_spot_list = [kernel_wt_list[entry][:,spt_num] for entry in range(len(kernel_wt_list))]

        # deconv_op, cell_wts, res1, alpha, debug_vars_list = run_adam_softmax_optimization_ms(Bscrna, X1_list,curr_slice_idx,# October2024 - fix kernel_wt[spt_num] -> kernel_wt[:,spt_num]
        deconv_op, cell_wts = run_adam_softmax_optimization_ms(Bscrna, X1_list,curr_slice_idx,# October2024 - fix kernel_wt[spt_num] -> kernel_wt[:,spt_num]
                                                                     kernel_wt_spot_list, 
                                                                     ct_identity,
                                                                     res1_init, 
                                             V_init=V_init, 
                                             lambda_=par_lambda,
                                             eta1_ = par_eta1, 
                                             max_iter_adam=max_iter_adam,
                                     # self_wt_=self_wt_,
                                     adj_wt_ = adj_wt_,
                                     par_lr_adam=par_lr_adam)
        
        deconv_df.loc[:,spt_num] = deconv_op.flatten() #.detach().cpu().numpy()
        cell_wts_df.loc[:,spt_num] = cell_wts.flatten()

        # debug only 
        # debug_vars_list_spots.append(debug_vars_list)
        
    return deconv_df, cell_wts_df # , res1, alpha, debug_vars_list_spots


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
                                adata_bulk_init=None, gt_align=True):

    # gt_align: If true, ground truth alignment is used. If False, alignment is computed using moscot
    # ensure same genes in ST, scRNA
    # Bsc: gene X cell matrix from reference 
    # ct_identity: cell X cell-type binary matrix 
    
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
    
    kernel_wt_list = []
    for entry in range(len(adata_ip)):
        if entry == curr_slice_idx:
            kernel_wt1 = get_gauss_kernel_wt(adata_ip[curr_slice_idx],spa_key1,min_wt=min_wt,bandwidth=bandwidth,
                                                        recompute=recompute,n_spatial_neigh=n_spatial_neigh,
                                                        nn_only=nn_only,weight_spatial=weight_spatial) 
        else:
            kernel_wt1 = np.zeros((adata_ip[entry].shape[0],adata_ip[curr_slice_idx].shape[0]))
            # print(adata_ip[entry].shape[0],adata_ip[curr_slice_idx].shape[0])
            np.fill_diagonal(kernel_wt1,0.75)
        
        # print(entry, kernel_wt1)
        kernel_wt_list.append(kernel_wt1)
    # print(kernel_wt_list)
    Bscrna = torch.tensor(np.array(Bsc), dtype=torch.float32,requires_grad=False).to(device)
    
    ct_identity = torch.tensor(np.array(ct_identity), dtype=torch.float32,requires_grad=False).to(device)

    adata_ip_x = [entry.X for entry in adata_ip]
    
    # deconv_st1, cell_wts, res1_1,alpha, debug_vars1 = get_adam_softmax_solution_perslice_ms(kernel_wt_list, curr_slice_idx,
    deconv_st1, cell_wts = get_adam_softmax_solution_perslice_ms(kernel_wt_list, curr_slice_idx,
                                                Bscrna,ct_identity,
                                            all_ctypes, adata_ip_x, 
                                            par_lambda=par_lambda, 
                                            par_eta1=par_eta1, 
                                            max_iter_adam=max_iter_adam,
                                            adj_wt_=adj_wt,
                                            ct_props=ct_props,
                                            par_lr_adam=par_lr_adam,
                                            adata_bulk_init=adata_bulk_init)
        
    return deconv_st1, cell_wts # , res1_1, alpha, debug_vars1


# def spadecoder_multislice(adata_ip,  # this is a list of slices (in order) with each key being a slice
#                                     Bref,ct_identity, spa_key='spatial',
#                                     min_wt=0.0001, renorm=True,recompute=True,
#                                                                 bandwidth=0.01,
#                                                                 par_lambda=2,par_eta1=10.0,
#                                 max_iter_adam=500,
#                                 n_spatial_neigh=15,nn_only=True,
#                                 kernel_combo_method='M1',par_lr_adam=0.01,
#                                 weight_transcr=0.0,weight_spatial=1.0,
#                                 ct_props=None, adj_wt=None, adata_bulk_init=None):

#     # loop through each slice and call spadecoder_slice_wrapper_ms
#     tissue_op_bulk = {}
#     cellwt_op_bulk = {}
#     res_op_bulk = {}
#     alpha_op_bulk = {}
#     debug_vars_bulk = {}


#     tissue_op_sc = {}
#     cellwt_op_sc = {}
#     res_op_sc = {}
#     alpha_op_sc = {}
#     debug_vars_sc= {}

#     for curr_slice_idx in range(len(adata_ip)):
#         tissue_op_bulk[curr_slice_idx],cellwt_op_bulk[curr_slice_idx], res_op_bulk[curr_slice_idx], alpha_op_bulk[curr_slice_idx], debug_vars_op_bulk[curr_slice_idx] = spadecoder_slice_wrapper_ms(adata_ip, # this is a list 
#                                 curr_slice_idx, 
#                                 Bref, ct_identity,
#                                 spa_key1='spatial',
#                                 min_wt=0.0001,
#                                 renorm=True,
#                                 bandwidth=0.01,
#                                 recompute=True,par_lambda=0.001, 
#                                 par_eta1=10.0,
#                                 max_iter_adam=500,
#                                 n_spatial_neigh=15,nn_only=True,
#                                 weight_spatial=1.0,
#                                 adj_wt=None,
#                                 ct_props=None,
#                                 par_lr_adam=0.01, 
#                                 adata_bulk_init=None)

#         deconv_st1_sc[key_name][entry3][entry0], _  = spadecoder_slice_wrapper(adata_spa[key_name][entry0][entry3], # adata_spot_swap[key_name][entry3], 
#                                                                 adata_sc_df,ct_identity_sc, #Palign='no neigh', 
#                                                                 min_wt=min_wt,
#                                                                 spa_key1=spa_key1,# spa_key2=spa_key2,
#                                                                 renorm=renorm,recompute=recompute,
#                                                                 bandwidth=bandwidth,
#                                                                 par_lambda=par_lambda,
#                                                         self_wt1=self_wt1,# self_wt2=self_wt2,
#                                                         # adj_wt1=adj_wt1,# adj_wt2=adj_wt2,
#                                 # admm_tau=admm_tau, #max_iter_admm=max_iter_admm, 
#                                 max_iter_adam=max_iter_adam,# tol=1e-4,
#                                 n_spatial_neigh=n_spatial_neigh,nn_only=nn_only,
#                                 n_transcr_neigh=n_transcr_neigh,n_pcs=n_pcs,
#                                 kernel_combo_method=kernel_combo_method,par_lr_adam=par_lr_adam,
#                                 weight_transcr=weight_transcr,weight_spatial=weight_spatial,
#                                 # transcr_kernel_thresh=transcr_kernel_thresh,
#                                 ct_props=None,
#                                 par_eta1=par_eta1,
#                                 adata_bulk_init=deconv_st1_bulk[key_name][entry3][entry0])                     

    
#     return tissue_op, cellwt_op, res_op, alpha_op, debug_vars_op




