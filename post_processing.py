from .importing_modules import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def morans_I_permutation(curr_slices,celltypes,  seed=0, nsim=1000, batch_sz=1000, sig_thresh=0.05 ):

    celltype_props = torch.as_tensor(np.array(curr_slices.obs[celltypes]),device=device)

    mean_celltype_props = celltype_props.mean(axis=0)

    celltype_props_diff = celltype_props - mean_celltype_props


    neigh_conn = torch.as_tensor(curr_slices.obsp['spatial_connectivities'], device=device, dtype=torch.float32)

    # print(neigh_conn.sum(axis=0))
    neigh_conn = neigh_conn.fill_diagonal_(1)

    morans_tmp = ((celltype_props_diff.T.unsqueeze(2) * celltype_props_diff.T.unsqueeze(1))  * neigh_conn.unsqueeze(0))
    # morans_denom = morans_tmp.diagonal(dim1=1, dim2=2).sum(axis=1)

    moran_norm_const = (neigh_conn.shape[0]/neigh_conn.sum())
    moransI = (morans_tmp.sum(axis=2).sum(axis=1) / morans_tmp.diagonal(dim1=1, dim2=2).sum(axis=1))*moran_norm_const


    # hypothesis testing 
    gen = torch.Generator(device=device).manual_seed(seed)

    nspots_src = curr_slices.shape[0]



    counts_both = torch.zeros(len(celltypes),device=device)
    counts = torch.zeros(len(celltypes),device=device)


    for start in range(0, nsim, batch_sz):
        end = min(start + batch_sz, nsim)
        batch_n = end - start

        rand_vals = torch.rand((batch_n, nspots_src), generator=gen, device=device)
        perm_ids     = rand_vals.argsort(dim=1) 
        
        celltype_props_diff_perm = torch.transpose(celltype_props_diff[perm_ids], 1,2)

        morans_tmp_perm = ((celltype_props_diff_perm.unsqueeze(3) * celltype_props_diff_perm.unsqueeze(2))  * neigh_conn.unsqueeze(0).unsqueeze(1))

        moransI_perm = (morans_tmp_perm.sum(axis=3).sum(axis=2) / morans_tmp_perm.diagonal(dim1=2, dim2=3).sum(axis=2))*moran_norm_const

        counts_both = counts_both + (moransI_perm.abs() > moransI.abs()).sum(axis=0) # /moransI_perm.shape[0]
        counts = counts + (moransI_perm > moransI).sum(axis=0) # /moransI_perm.shape[0]

    counts = counts/nsim
    counts_both = counts_both/nsim
    sig_celltypes_both = np.array(celltypes)[(counts_both.detach().cpu() < sig_thresh).nonzero(as_tuple=True)[0]]
    sig_celltypes = np.array(celltypes)[(counts.detach().cpu() < sig_thresh).nonzero(as_tuple=True)[0]]

    return moransI.detach().cpu(), counts.detach().cpu(), counts_both.detach().cpu(), sig_celltypes, sig_celltypes_both





def local_morans_I_permutation(curr_slices,celltypes,  seed=0, nsim=1000, batch_sz=1000, sig_thresh=0.05 ):

    celltype_props = torch.as_tensor(np.array(curr_slices.obs[celltypes]), device=device)

    mean_celltype_props = celltype_props.mean(axis=0)

    celltype_props_diff = celltype_props - mean_celltype_props


    neigh_conn = torch.as_tensor(curr_slices.obsp['spatial_connectivities'], device=device, dtype=torch.float32)

    # print(neigh_conn.sum(axis=0))
    neigh_conn = neigh_conn.fill_diagonal_(1)

    nspots_src, n_celltypes = celltype_props_diff.shape

    local_morans_tmp = ((celltype_props_diff.T.unsqueeze(2) * celltype_props_diff.T.unsqueeze(1))  * neigh_conn.unsqueeze(0))
    morans_denom = local_morans_tmp.diagonal(dim1=1, dim2=2).sum(axis=1)
    morans_denom_loop = (nspots_src/ morans_denom).unsqueeze(0).unsqueeze(1)
    # moran_norm_const = (neigh_conn.shape[0]/neigh_conn.sum())
    local_moransI = (local_morans_tmp.sum(axis=1) / morans_denom.unsqueeze(1))* neigh_conn.shape[0] # moran_norm_const

    # hypothesis testing 
    gen = torch.Generator(device=device).manual_seed(seed)

    counts = torch.zeros((nspots_src,n_celltypes),device=device)
    counts_both = torch.zeros((nspots_src,n_celltypes),device=device)

    # nsim_total = nsim * nspots_src
    # batch_sz_total = batch_sz * nspots_src

    full = torch.arange(nspots_src)           # shape (n,)
    grid = full.repeat(nspots_src, 1)                        # shape (n, n)

    # Step 2: Mask out diagonal (i == j)
    mask = torch.eye(nspots_src, dtype=torch.bool)
    leave_one_out = grid[~mask].view(nspots_src, nspots_src - 1)      # shape (n, n-1)
    leave_one_out = leave_one_out.to(device)
    
    mask = ~torch.eye(nspots_src, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # (S, S) -> # (1, S, S, 1)
    mask = mask.to(device)

    celltype_props_diff_sq = (celltype_props_diff*celltype_props_diff).unsqueeze(0) # (S, C) -> (1, S, C)

    W_perm = neigh_conn.T[torch.arange(nspots_src).unsqueeze(1), leave_one_out]  # (S, S-1)
    W_perm = W_perm.unsqueeze(0).unsqueeze(-1)  # (1, S, S-1, 1)
    
    for start in range(0, nsim, batch_sz):
        end = min(start + batch_sz, nsim)
        batch_n = end - start

        rand_vals = torch.rand((batch_n,nspots_src, nspots_src-1), generator=gen, device=device)
        perm_ids     = rand_vals.argsort(dim=2) 

        permuted_spot_ids = torch.gather(leave_one_out.unsqueeze(0).expand(batch_n, -1, -1), dim=2, index=perm_ids)

        
        celltype_props_diff_expand = celltype_props_diff.unsqueeze(0).unsqueeze(1).expand(batch_n,nspots_src,nspots_src,-1)

        permuted_spot_ids_expand = permuted_spot_ids.unsqueeze(3).expand(-1, -1, -1, n_celltypes)

        permuted_ct = torch.gather(celltype_props_diff_expand, dim=2, index=permuted_spot_ids_expand)  # (batch_n, S, S-1, C)

        
        # Weighted sum over permuted neighbors: (batch_n, S, C)
        permuted_ct_weighted_sum = (permuted_ct * W_perm).sum(dim=2)
        # Original z_i: (1, S, C)
        celltype_props_diff_i = celltype_props_diff.unsqueeze(0)
        # Compute local Moran's I: (P, S, C)
        local_moran_perm = (celltype_props_diff_i * permuted_ct_weighted_sum) + celltype_props_diff_sq
        local_moran_perm = local_moran_perm*morans_denom_loop

        counts_both = counts_both + (local_moran_perm.abs() > local_moransI.T.abs().unsqueeze(0)).sum(axis=0) # /moransI_perm.shape[0]
        counts = counts + (local_moran_perm > local_moransI.T.unsqueeze(0)).sum(axis=0) # /moransI_perm.shape[0]

    counts = counts/nsim
    counts_both = counts_both/nsim

    signed_local_morans = local_moransI.T * torch.sign(celltype_props_diff) 
    
    is_sig = (counts < sig_thresh).int()
    is_sig_both = (counts_both < sig_thresh).int()
    
    return local_moransI.T.detach().cpu(), signed_local_morans.detach().cpu(), is_sig.detach().cpu(), is_sig_both.detach().cpu(), counts.detach().cpu(), counts_both.detach().cpu()
    # sig_celltypes = np.array(celltypes)[(counts < sig_thresh).nonzero(as_tuple=True)[0]]




def sci_permutation(curr_slices,celltypes,  seed=0, nsim=1000, batch_sz=1000, sig_thresh=0.05):

    celltype_props = torch.as_tensor(np.array(curr_slices.obs[celltypes]), device=device)

    mean_celltype_props = celltype_props.mean(axis=0)

    celltype_props_diff = celltype_props - mean_celltype_props


    neigh_conn = torch.as_tensor(curr_slices.obsp['spatial_connectivities'], device=device,dtype=torch.float32)

    # print(neigh_conn.sum(axis=0))
    neigh_conn = neigh_conn.fill_diagonal_(1.0)

    # global sci
    nspots_src = curr_slices.shape[0]
    n_celltypes = len(celltypes)


    # prod_neigh_prop = ((celltype_props_diff.unsqueeze(1).unsqueeze(3) * celltype_props_diff.unsqueeze(0).unsqueeze(2)) * neigh_conn.unsqueeze(2).unsqueeze(3))
    # sci_num = prod_neigh_prop.sum(axis=0).sum(axis=0)
    # T = celltype_props_diff.T  # K x S
    neigh_T = torch.matmul(celltype_props_diff.T, neigh_conn)  # K x S
    sci_num = torch.matmul(neigh_T, celltype_props_diff)  # K x K

    # denom_tmp = ((prod_neigh_prop.diagonal(dim1=0,dim2=1)).diagonal(dim1=0, dim2=1)).sum(axis=0).sqrt().unsqueeze(0)
    # denom = denom_tmp * denom_tmp.T
    diag_terms = (celltype_props_diff * celltype_props_diff).sum(axis=0)  # K
    denom = torch.sqrt(torch.ger(diag_terms, diag_terms))  # K x K
    numerator_norm_const = nspots_src / (2 * neigh_conn.sum())
    mult_factor = numerator_norm_const / denom # C x C 
    sci_global = sci_num * mult_factor # C x C 


    ### two sided permutation testing 
    gen = torch.Generator(device=device).manual_seed(seed)

    counts = torch.zeros(n_celltypes,n_celltypes, device=device)
    counts_both = torch.zeros(n_celltypes,n_celltypes, device=device)

    for start in range(0, nsim, batch_sz):
        end = min(start + batch_sz, nsim)
        batch_n = end - start

        rand_vals = torch.rand((batch_n, nspots_src), generator=gen, device=device)
        perm_ids     = rand_vals.argsort(dim=1) 
        
        celltype_props_diff_perm = celltype_props_diff[perm_ids]

        permuted_T = celltype_props_diff_perm.transpose(1, 2) # P K S 
        # print(permuted_T.dtype, neigh_conn.dtype)
        neigh_prod = torch.matmul(permuted_T, neigh_conn) # sum across neigh -> P K S 
        sci_num_perm = torch.matmul(neigh_prod, celltype_props_diff_perm) # P K S x P S K -> P K K  
        
        # prod_neigh_prop_perm = ((celltype_props_diff_perm.unsqueeze(2).unsqueeze(4) * celltype_props_diff_perm.unsqueeze(1).unsqueeze(3)) *  neigh_conn.unsqueeze(2).unsqueeze(3).unsqueeze(0))
        # sci_num_perm = prod_neigh_prop_perm.sum(axis=1).sum(axis=1) # P x C x C 

        sci_global_perm = sci_num_perm * mult_factor.unsqueeze(0)

        counts_both = counts_both + (sci_global_perm.abs() > sci_global.abs()).sum(axis=0) # /moransI_perm.shape[0]
        counts = counts + (sci_global_perm > sci_global).sum(axis=0) # /moransI_perm.shape[0]

    counts = counts/nsim
    counts_both = counts_both/nsim
    sig_celltype_combos_both  = pd.DataFrame((counts_both < sig_thresh).int().detach().cpu().numpy(), index=celltypes, columns = celltypes)
    sig_celltype_combos  = pd.DataFrame((counts < sig_thresh).int().detach().cpu().numpy(), index=celltypes, columns = celltypes)
    # here  C C -> current spot celltype is at dim 1 (not 0)
    return sci_global.detach().cpu(), counts.detach().cpu(), counts_both.detach().cpu(), sig_celltype_combos, sig_celltype_combos_both



def local_sci_permutation(curr_slices,celltypes,  seed=0, nsim=1000, batch_sz=1000, sig_thresh=0.05):

    celltype_props = torch.as_tensor(np.array(curr_slices.obs[celltypes]), device=device)

    mean_celltype_props = celltype_props.mean(axis=0)

    celltype_props_diff = celltype_props - mean_celltype_props


    neigh_conn = torch.as_tensor(curr_slices.obsp['spatial_connectivities'], device=device, dtype=torch.float32)

    # print(neigh_conn.sum(axis=0))
    neigh_conn = neigh_conn.fill_diagonal_(1)

    nspots_src, n_celltypes = celltype_props_diff.shape

    weighted_neigh = torch.matmul(celltype_props_diff.T, neigh_conn) # C x S 
    sci_local_num = weighted_neigh.T.unsqueeze(1) * celltype_props_diff.unsqueeze(2) # S x C x C 
    # prod_neigh_prop = ((celltype_props_diff.unsqueeze(1).unsqueeze(3) * celltype_props_diff.unsqueeze(0).unsqueeze(2)) * neigh_conn.unsqueeze(2).unsqueeze(3))
    # sci_local_num = prod_neigh_prop.sum(axis=0)
    diag_terms = (celltype_props_diff * celltype_props_diff).sum(axis=0)  # K
    denom = torch.sqrt(torch.ger(diag_terms, diag_terms))  # K x K

    #denom_tmp = ((prod_neigh_prop.diagonal(dim1=0,dim2=1)).diagonal(dim1=0, dim2=1)).sum(axis=0).sqrt().unsqueeze(0)
    #denom = denom_tmp * denom_tmp.T
    numerator_norm_const = nspots_src / neigh_conn.sum()
    mult_factor = (numerator_norm_const / denom).unsqueeze(0) # C x C 
    sci_local = sci_local_num * mult_factor
    mult_factor = mult_factor.unsqueeze(0) # 1 x S X C X C 


    # hypothesis testing 
    gen = torch.Generator(device=device).manual_seed(seed)

    counts = torch.zeros((nspots_src,n_celltypes,n_celltypes),device=device)
    counts_both = torch.zeros((nspots_src,n_celltypes,n_celltypes),device=device)

    full = torch.arange(nspots_src)           # shape (n,)
    grid = full.repeat(nspots_src, 1)                        # shape (n, n)

    # Step 2: Mask out diagonal (i == j)
    mask = torch.eye(nspots_src, dtype=torch.bool)
    leave_one_out = grid[~mask].view(nspots_src, nspots_src - 1)      # shape (n, n-1)
    leave_one_out = leave_one_out.to(device)

    mask = ~torch.eye(nspots_src, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # (S, S) -> # (1, S, S, 1)
    mask = mask.to(device)

    celltype_props_diff_sq = (celltype_props_diff.unsqueeze(2) *celltype_props_diff.unsqueeze(1)).unsqueeze(0) # S, C, 1 * S, 1, C -> S x C x C -> 1 X S X C X C  

    W_perm = neigh_conn.T[torch.arange(nspots_src).unsqueeze(1), leave_one_out].unsqueeze(0).unsqueeze(-1) # 1 X S X S -1 X 1

    for start in range(0, nsim, batch_sz):
        end = min(start + batch_sz, nsim)
        batch_n = end - start

        rand_vals = torch.rand((batch_n,nspots_src, nspots_src-1), generator=gen, device=device)
        perm_ids     = rand_vals.argsort(dim=2) 

        permuted_spot_ids = torch.gather(leave_one_out.unsqueeze(0).expand(batch_n, -1, -1), dim=2, index=perm_ids)  # P, S, S-1

        celltype_props_diff_expand = celltype_props_diff.unsqueeze(0).unsqueeze(1).expand(batch_n,nspots_src,nspots_src,-1) # [P, S, S, C]

        permuted_spot_ids_expand = permuted_spot_ids.unsqueeze(3).expand(-1, -1, -1, n_celltypes) # P, S, S-1, C

        permuted_ct = torch.gather(celltype_props_diff_expand, dim=2, index=permuted_spot_ids_expand)  # (batch_n, S, S-1, C)

        # W_perm = neigh_conn[torch.arange(nspots_src).unsqueeze(1), leave_one_out]  # (S, S-1)
        # W_perm = W_perm.unsqueeze(0).unsqueeze(-1)  # (1, S, S-1, 1)
        
        # Weighted sum over permuted neighbors: (P, S, C)
        permuted_ct_weighted_sum = (permuted_ct * W_perm).sum(dim=2)
        
        local_sci_tmp = (permuted_ct_weighted_sum.unsqueeze(3) * celltype_props_diff.unsqueeze(0).unsqueeze(2)) + celltype_props_diff_sq  # P, S, C, C + 1, S, C, C 

        local_sci_perm = local_sci_tmp * mult_factor

        counts_both = counts_both + (local_sci_perm.abs() > sci_local.abs().unsqueeze(0)).sum(axis=0) # /moransI_perm.shape[0]
        counts = counts + (local_sci_perm > sci_local.unsqueeze(0)).sum(axis=0)

    counts = counts/nsim
    counts_both = counts_both/nsim

    is_sig = (counts < sig_thresh).int()
    is_sig_both = (counts_both < sig_thresh).int()

    # here S C C -> current spot celltype is at dim 1 (not 0 or 2)
    return sci_local.detach().cpu(),  is_sig.detach().cpu(), is_sig_both.detach().cpu(), counts.detach().cpu(), counts_both.detach().cpu()