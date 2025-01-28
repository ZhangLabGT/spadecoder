from importing_modules import *
# from simulations import *
from processing_for_model import *

import logging
import os 
from datetime import datetime

# datetime object containing current date and time
# now = datetime.now()

# dir_path = os.path.dirname(os.path.realpath(__file__))
# logging.basicConfig(filename=dir_path + '/logging/spadecoder_singlecell_with_adam_softmax_' + str(now) + '.log',
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)


# logging.info("Spadecoder Pytorch No Bias ADMM + Adam")

# logger = logging.getLogger('SpaDecoder Pytorch  No Bias ADMM + Adam')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# logging.info(f"Using device: {device}")


####################   use other tools to align adjacent slices #############
def solve_alignment(adata_sp):
    ap = AlignmentProblem(adata=adata_sp)
    ap = ap.prepare(batch_key="batch", policy="sequential")
    ap = ap.solve()
    
    return ap
############################################################################




# optional to check convergence
def compute_cost_function(V, res1, alpha, X1, 
                          # X2, 
                          Bf, w, # P, 
                            ct_identity, lambda_,
                           eta1_, 
                           # eta2_, # Yconst,  muConst, 
                                        self_wt_=1.0,# adj_wt_=1.0,#tau=1.0
                                        ):
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
    
    # B = torch.tensor(B, dtype=torch.float32).to(device)  # Ensure B is a PyTorch tensor
    B_prime = Bf #+ Bvar #(bias.expand(-1,B.shape[0]).T) # .to(device)
    # w = torch.tensor(w, dtype=torch.float32).to(device)  # Ensure w is a PyTorch tensor
    # P = torch.tensor(np.array(P), dtype=torch.float32).to(device)  # Ensure P is a PyTorch tensor

    # res1_softmax = torch.nn.functional.softmax(res1,dim=0)
    # torch.nn.functional.softmax(res1,dim=0)
    # why are we applying softmax to columns of res 1 ? -> ct_identity is column normalized i.e. each col sums to 1 so it's like "averaging"
    # hence since we're keeping res1 close to ct_identity, we need to column normalize this too 
    bv_prod_X1 =  B_prime @  torch.nn.functional.softmax(res1,dim=0) @  torch.nn.functional.softmax(V, dim=0) #bv_prod.expand(-1, X1.shape[1])
    
    batch_id = (w != 0).squeeze()
    X1_batch = X1[:, batch_id]
    
    # print(X1_batch.shape)

    # G x N 
    bv_prod_X1 = bv_prod_X1.expand(-1, X1_batch.shape[1])
    
    # bv_prod_X2 = B_prime @ (res2 + ct_identity) @  torch.nn.functional.softmax(V, dim=0) #bv_prod.expand(-1, X2.shape[1])
    
    # bv_prod_X1 =  B_prime @ ( ct_identity) @  torch.nn.functional.softmax(V, dim=0) #bv_prod.expand(-1, X1.shape[1])
    # bv_prod_X2 = B_prime @ ( ct_identity) @  torch.nn.functional.softmax(V, dim=0) #bv_prod.expand(-1, X2.shape[1])
    
    term1 = self_wt_ *  0.5 * (w[batch_id] * (X1_batch -  torch.clamp(alpha, min = 1e-4) * bv_prod_X1).pow(2).sum(0)).mean()
    # Compute the first term: sum of weighted squared L2 norms for X1
    
    # term1 = self_wt_ *  0.5*torch.sum(w * torch.norm(X1 -  bv_prod_X1,p=2, dim=0)**2)
    # term2 = adj_wt_ *  0.5*torch.sum(P * torch.norm(X2 - bv_prod_X2, p=2, dim=0)**2)
    
    # Compute the third term: L2 regularization term
    #l1_term = alpha_ * lambda_ * torch.sum(torch.abs(V))

    # Compute the third term: L2 regularization term
    l2_term = 0.5 *  lambda_ * (V.pow(2).sum())

    # res1_term = 0
    res1_term = 0.5 * eta1_ * ((torch.nn.functional.softmax(res1,dim=0)-ct_identity).pow(2).mean()) # (torch.nn.functional.softmax(res1,dim=0) - ct_identity).pow(2)
    # 0.5 * eta1_ * torch.sum(torch.norm(res1, p=2, dim=None))


    # move to cross entropy loss
    # loss = torch.nn.CrossEntropyLoss()
    # res1_term = eta1_ * loss(torch.nn.functional.softmax(res1), ct_identity) # 0.5 * eta1_ *  ((torch.nn.functional.softmax(res1,dim=0)-ct_identity).pow(2).mean())
    


    # res2_term = 0.5 * eta2_ * torch.sum(torch.norm(res2, p=2, dim=None))
    # Total cost
    cost = term1 +  l2_term + res1_term # term2 +   + res2_term

    # # augmented lagrangian
    # aug_lagrangian = cost + (tau / 2) * torch.norm((V - Yconst + muConst),p=2,dim=0)**2 #+ checkY(Yconst)

    #(tau / 2) * torch.sum((V - Yconst + muConst)**2) + (rho / 2) * torch.sum((Bvar - Zconst + etaConst)**2)
    
    return cost # , term1,  l2_term, res1_term # term2,  , ,  res2_term




def run_adam_softmax_optimization(Bf,  X1, # X2,  
                                  w, #P, 
                                  ct_identity, res1_init, # res2_init,
                                  V_init,
                       lambda_, eta1_, #eta2_, # alpha_,
                        # Yconst,  muConst, # etaConst, 
                        self_wt_=1.0,#adj_wt_=1.0, 
                        max_iter_adam=500, #printiter=1, 
                        par_lr_adam=1e-2 #, dist_measure='L2',
                        #tau=1.0,rho=1.0,
                        #logging_active=False
                        ):
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
    # Bnet = Bf + Bvar_init

    V = V_init.detach().clone().requires_grad_(True).to(device) # torch.tensor(V_init, requires_grad=True, dtype=torch.float32).to(device)
    #Bvar = Bvar_init.detach().clone().requires_grad_(True).to(device) # torch.tensor(Bvar_init, requires_grad=True, dtype=torch.float32).to(device)
    res1 = res1_init.detach().clone().requires_grad_(True).to(device) # torch.tensor(Bvar_init, requires_grad=True, dtype=torch.float32).to(device)
    # res2 = res2_init.detach().clone().requires_grad_(True).to(device) # torch.tensor(Bvar_init, requires_grad=True, dtype=torch.float32).to(device)
    alpha = torch.tensor([1.0], requires_grad=True, device=device)
    optimizer = torch.optim.Adam([V, res1, alpha], lr=par_lr_adam)

    
    cost_list = []
    # term1_list = []
    # # term2_list = []
    # l2_term_list = []
    # res1_term_list = []
    # alpha_list = []
    # res1_softmax_list = []
    # res2_term_list = []
    #print(V.size())
    for iteration in range(max_iter_adam):
        optimizer.zero_grad()
        
        cost  = compute_cost_function(V, res1,alpha, X1, #X2, 
                                                                 Bf, w, #P, 
                                                                 ct_identity, lambda_,eta1_, #eta2_,  
                                                                                self_wt_=self_wt_) #,adj_wt_=adj_wt_)
        
        
        cost_list.append(cost)
        #term1_list.append(term1)
        # term2_list.append(term2)
        #l2_term_list.append(l2_term)
        #res1_term_list.append(res1_term)
        # alpha_list.append(alpha_term)
        # res1_softmax_list.append(res1_softmax)
        # res2_term_list.append(res2_term)

        if len(cost_list) >= 2: 
            if torch.abs(cost_list[-2] -  cost_list[-1]) < 0.0001:
                break

        # I removed the F2Y term from the augmented lagrangian
        cost.backward()
        optimizer.step()
        
        # if (iteration % printiter == 0) and logging_active:
        #     # logging.INFO(f"Iteration {iteration}, Cost: {cost.item()}")
        #     logging.info(f"ADAM Iteration {iteration},  AugLagCost: {total_cost},  Cost: {cost}, \n\n SameSliceTerm: {term1}, NbrSliceTerm: {term2}, L1Term: {l1_term}, L2Term:{l2_term}\n\n")
        #     logging.info("deconv:" + ', '.join([str(entry) for entry in list(V.data.cpu().numpy().T)]) + '\n\n')   
        #     logging.info("Bvar:" + ', '.join([str(entry) for entry in list(Bvar.data.cpu().numpy().T)]) + '\n\n')   
        

        #print(V.size())

        # probably add a stopping criteria here too 

    # apply softmax 
    V = torch.nn.functional.softmax(V, dim=0) # without dim=0, the outputs was all 1s
    res1 = torch.nn.functional.softmax(res1,dim=0)
    cell_wts = res1 @ V
    # all_debug_vars = (cost_list,term1_list,l2_term_list,res1_term_list)

    return V.detach().cpu().numpy(), cell_wts.detach().cpu().numpy() # ,res1.detach().cpu().numpy(), alpha.detach().cpu().numpy() # , all_debug_vars  # , Bvar.detach()




   


def get_adam_softmax_solution_perslice(#Palign,
                                    kernel_wt, Bscrna, ct_identity,
                                    all_ctypes, expr_st1, #expr_st2, # deconv_init,
                                    par_lambda=1.0, 
                                    par_eta1=5.0, #par_eta2 = 5.0, 
                                    # max_iter_admm=10, 
                                    max_iter_adam=500, # tol=1e-4,
                      self_wt_=1.0,#adj_wt_=1.0,
                      ct_props=None,
                      #printiter=100,gradclip_maxnorm=1.0,
                      par_lr_adam=1e-2,adata_bulk_init = None
                      #gradclip=True,
                      #dist_measure='L2',#logging_active=False # ,optim_strategy='adam'
                      ):
    
    n_celltypes = len(all_ctypes)
    n_spots_st1 = expr_st1.shape[0]
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

        V_init = np.array(V_init).reshape(-1,1)

        V_init = torch.tensor(V_init,  dtype=torch.float32,requires_grad=False).to(device)

    
    X1 = torch.tensor(expr_st1.T, dtype=torch.float32,requires_grad=False).to(device)  # Ensure X1 is a PyTorch tensor
    # X2 = torch.tensor(expr_st2.T, dtype=torch.float32,requires_grad=False).to(device)  # Ensure X2 is a PyTorch tensor
    
    # V_init = torch.tensor(np.array(deconv_init.T),  dtype=torch.float32).to(device)
    
    kernel_wt = torch.tensor(kernel_wt, dtype=torch.float32,requires_grad=False).to(device)  # Ensure w is a PyTorch tensor
    # Palign = torch.tensor(np.array(Palign), dtype=torch.float32,requires_grad=False).to(device)  # Ensure P is a PyTorch tensor
    
    # Bscrna = torch.tensor(Bscrna, dtype=torch.float32).to(device)
    # Bnet_min = torch.tensor(Bnet_min, dtype=torch.float32).to(device) # bscrna-bstd 
    # Bnet_max = torch.tensor(Bnet_max, dtype=torch.float32).to(device)
    # debug_vars_list_spots = []

    # initlaize same as ct-identity so close to softmax is close to ctype 
    res1_init = torch.log(ct_identity) # this gets to infiniy but with the softmax, it's fine
    # print(res1_init)
    
    for spt_num in range(n_spots_st1):  
        # self_wt_ = self_wt_ * (1/torch.sum(kernel_wt[spt_num]))
        # if torch.sum(Palign[spt_num]) == 0: # in case no adj slice, set 1 
        #     adj_wt_ = 1.0
        # else:
        #     adj_wt_ = adj_wt_ * (1/torch.sum(Palign[spt_num]))

        # 0-initialization
        # res1_init =  torch.zeros(size=(Bscrna.shape[1],n_celltypes), dtype=torch.float32,requires_grad=False,device=device)
        
        
        # res2_init =  torch.zeros(size=(Bscrna.shape[1],n_celltypes), dtype=torch.float32,requires_grad=False,device=device)
        if adata_bulk_init is not None:
            #print(adata_bulk_init)
            V_init = np.array(adata_bulk_init.loc[all_ctypes,str(spt_num)]).reshape(-1,1)
            V_init = torch.tensor(V_init,dtype=torch.float32,requires_grad=False).to(device)

        deconv_op, cell_wts = run_adam_softmax_optimization(Bscrna, X1, #X2, 
                                                                                # October2024 - fix kernel_wt[spt_num] -> kernel_wt[:,spt_num]
                                                                     kernel_wt[:,spt_num], #Palign[spt_num], 
                                                                     ct_identity,
                                                                     res1_init, # res2_init,
                                             V_init=V_init, 
                                             lambda_=par_lambda,
                                             eta1_ = par_eta1, #eta2_ = par_eta2, #tau=tau, 
                                             # max_iter_admm=max_iter_admm,
                                             max_iter_adam=max_iter_adam,
                                     self_wt_=self_wt_, #adj_wt_=adj_wt_,
                                     par_lr_adam=par_lr_adam,# dist_measure=dist_measure,# gradclip=True,
                      # logging_active=logging_active)
                                     #printiter=printiter,# gradclip_maxnorm=gradclip_maxnorm,
                      #par_lr=par_lr,gradclip=gradclip,dist_measure=dist_measure,
                      #logging_active=logging_active)
        )
        


        deconv_df.loc[:,spt_num] = deconv_op.flatten() #.detach().cpu().numpy()
        cell_wts_df.loc[:,spt_num] = cell_wts.flatten()
        # for debugging 
        # debug_vars_list_spots.append(debug_vars_list)

        # initialize proportions with the closest spot already predicted 
        # V_init = np.array(deconv_op.flatten().detach().cpu().numpy()).reshape(-1,1)
        # alternatively initialize all randomly 
    # debug_vars = (V_all_spots, cost_list_spots, aug_lag_list_spots, term1_list_spots, term2_list_spots, l2_term_list_spots)
    
    return deconv_df, cell_wts_df # ,res1, alpha, debug_vars_list_spots


def adam_softmax_solution_perslice_singlecellref_wrapper(adata_st1, #adata_st2, 
                                                         Bsc, ct_identity,
                                   spa_key1='spatial',# spa_key2='spatial_warped',
                  min_wt=0.0001,#Palign=None,
                  renorm=True,
                  bandwidth=0.01,
                  recompute=True,par_lambda=0.001, 
                  par_eta1=10.0,# par_eta2=5.0,
                  #admm_tau=1.0, # max_iter_admm=10, 
                  max_iter_adam=500,# tol=1e-4,
                  n_spatial_neigh=15,nn_only=True,
                  n_transcr_neigh=15,n_pcs=20,
                  kernel_combo_method='M1',
                  weight_transcr=0.0,weight_spatial=1.0,
                  self_wt1=1.0,# adj_wt1=1.0,# self_wt2=1.0, adj_wt2=1.0,
                  ct_props=None,
                  # printiter=100, transcr_kernel_thresh=0.6,
                  par_lr_adam=0.01, adata_bulk_init=None
                  #gradclip_maxnorm=1.0,par_lr=1e-4,gradclip=True,
                  #dist_measure='L2',
                  # logging_active=False,optim_strategy='adam'
                  ):
    # this version uses a spatio-transcriptomic kernel
    # inputs 
    # the 2 spots samples 
    # the avg scRNA expression across cell-types 
    # the scRNA object 
    
    # ensure same genes in ST, scRNA


    # Bsc: gene X cell matrix from reference 
    # ct_identity: cell X cell-type binary matrix 
    #  
    genes_to_use = list(set(adata_st1.var.index).intersection(set(Bsc.index)))
    
    adata_st1 = adata_st1[:,genes_to_use].copy()
    # adata_st2 = adata_st2[:,genes_to_use].copy()
    Bsc = Bsc.loc[genes_to_use,:].copy()
    
    # print("here")
    # add zeros if cell-types not present
    # cell-types restricted by what's in reference
    all_ctypes = list(ct_identity.columns) # .union(set(adata_st1.obs.columns).union(set(adata_st2.obs.columns))))
    # if set(B.columns) != set(all_ctypes):
    #     # add extra columns with 0s
    #     for entry in all_ctypes:
    #         if entry not in B.columns:
    #             B[entry] = 0.0
    
    if  set(adata_st1.obs.columns) != set(all_ctypes):
        # add extra columns with 0s
        for entry in all_ctypes:
            if entry not in set(adata_st1.obs.columns):
                obs =  sc.get.obs_df(adata_st1,keys=list(adata_st1.obs.columns))
                obs[entry] = 0.0
                adata_st1.obs = obs

    # if  set(adata_st2.obs.columns) != set(all_ctypes):
    #     # add extra columns with 0s
    #     for entry in all_ctypes:
    #         if entry not in set(adata_st2.obs.columns):
    #             obs =  sc.get.obs_df(adata_st2,keys=list(adata_st2.obs.columns))
    #             obs[entry] = 0.0
    #             adata_st2.obs = obs       

     
    kernel_wt1 = combine_transcr_spatial_kernel(adata_st1,spa_key1,min_wt=min_wt,bandwidth=bandwidth,
                                                recompute=recompute,n_spatial_neigh=n_spatial_neigh,
                                                nn_only=nn_only,n_transcr_neigh=n_transcr_neigh,n_pcs=n_pcs,
                                                kernel_combo_method=kernel_combo_method,weight_transcr=weight_transcr,
                                                weight_spatial=weight_spatial,
                                                #transcr_kernel_thresh = transcr_kernel_thresh
                                                ) 
    

    #logging.info(kernel_wt1)
    #logging.info('\n')
    # kernel_wt2 = combine_transcr_spatial_kernel(adata_st2,spa_key2,min_wt=min_wt,bandwidth=bandwidth,
    #                                             recompute=recompute,n_spatial_neigh=n_spatial_neigh,
    #                                             nn_only=nn_only,n_transcr_neigh=n_transcr_neigh,n_pcs=n_pcs,
    #                                             kernel_combo_method=kernel_combo_method,weight_transcr=weight_transcr,
    #                                             weight_spatial=weight_spatial,
    #                                             #transcr_kernel_thresh = transcr_kernel_thresh
    #                                             ) 
    # get P (alignment matrix), rows in slice st1, columns is st2
    # here P is the ground truth 
    # if Palign is None:
    #     Palign = np.diag(np.ones(adata_st1.shape[0])) # (adata_st1.shape[0],adata_st1.shape[0]))
    # elif Palign == 'no neigh':
    #     # initialize as zeros so neighboring slice won't be considered in deconvolution
    #     Palign = np.zeros((adata_st1.shape[0],adata_st2.shape[0]))
        

    # if Palign is None:
    #     # for no neighbor in adjacent slice
    #     # Palign1 = np.diag(np.ones(adata_st1.shape[0])) # (adata_st1.shape[0],adata_st1.shape[0])) 
    #     # Palign2 = np.diag(np.ones(adata_st1.shape[0]))
    #     # # using neighborhood in adjacent slice
    #     Palign1 = kernel_wt2.copy() # change this later
    #     Palign2 = kernel_wt1.copy()
    
    # elif Palign == 'no neigh':
    #     # initialize as zeros so neighboring slice won't be considered in deconvolution
    #     Palign1 = np.zeros((adata_st1.shape[0],adata_st2.shape[0]))
    #     Palign2 = np.zeros((adata_st2.shape[0],adata_st1.shape[0]))

    # else: # run with PASTE
    #     Palign1 = Palign.copy()
    #     Palign2 = Palign.T.copy()

    
    # Bscrna = np.array(B)
    Bscrna = torch.tensor(np.array(Bsc), dtype=torch.float32,requires_grad=False).to(device)
    # Bnet_min = torch.tensor(Bnet_min, dtype=torch.float32).to(device) # bscrna-bstd 
    # Bnet_max = torch.tensor(Bnet_max, dtype=torch.float32).to(device)
    # Bstd = np.array(Bstd)
    ct_identity = torch.tensor(np.array(ct_identity), dtype=torch.float32,requires_grad=False).to(device)
    

    deconv_st1, cell_wts = get_adam_softmax_solution_perslice(kernel_wt1, Bscrna,ct_identity,
                                            all_ctypes, adata_st1.X, # adata_st2.X, 
                                         #deconv_st1_init,
                                    par_lambda=par_lambda, 
                                    par_eta1=par_eta1, # par_eta2 = par_eta2, #tau=admm_tau,  
                                    # max_iter_admm=max_iter_admm, 
                                    max_iter_adam=max_iter_adam,
                                    self_wt_=self_wt1,# adj_wt_=adj_wt1,
                                    ct_props=ct_props,par_lr_adam=par_lr_adam,adata_bulk_init=adata_bulk_init)
                      

    # deconv_st2,res2_1,res2_2,  debug_vars2 = get_adam_softmax_solution_perslice(Palign2,kernel_wt2, Bscrna,ct_identity,  
    #                                         all_ctypes, adata_st2.X, adata_st1.X, 
    #                                      #deconv_st1_init,
    #                                 par_lambda=par_lambda, 
    #                                 par_eta1=par_eta1, par_eta2 = par_eta2, #tau=admm_tau,  
    #                                 # max_iter_admm=max_iter_admm,
    #                                 max_iter_adam=max_iter_adam,
    #                                 self_wt_=self_wt2,adj_wt_=adj_wt2,ct_props=ct_props,par_lr_adam=par_lr_adam)
    
   
    # if renorm:
    #     # print("ERROR. why did you need renormalization?")
    #     # print(deconv_st1, deconv_st2)
    #     # make deconv wts into propotions 
    #     deconv_st1[deconv_st1<0] = 0
    #     deconv_st1 = (deconv_st1/deconv_st1.sum())
        
    #     deconv_st2[deconv_st2<0] = 0
    #     deconv_st2 = (deconv_st2/deconv_st2.sum())
        
        
    return deconv_st1, cell_wts # , res1_1, alpha, debug_vars1

