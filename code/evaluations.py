from .importing_modules import *



def eval_deconv3(gnd_truth,deconv_res):
    ### evaluation of deconvolution results 
    # make sure gnd_truth and deconv_res have the same order of cell-names, even if the names differ 
    
    # input:
    # 1. ground truth -> cell by type df
    # 2. deconvolution results -> type by cell df
    # gnd_truth = gnd_truth_ip.copy()
    # deconv_res = deconv_res_ip.copy()
    
    gnd_truth = gnd_truth.T
    
    if gnd_truth.shape[0] != deconv_res.shape[0]:
        # add a ground truth row with all zero columns
        for entry in list(set(deconv_res.index) - set(gnd_truth.index)):
            gnd_truth.loc[entry,] = 0
    
    pred_dom_ct = deconv_res.idxmax()
    gt_dom_ct = gnd_truth.idxmax()
    
    
    deconv_res.columns = gnd_truth.columns # keeping cell ids consistent 
    
    assert set(deconv_res.index) == set(gnd_truth.index), "the ground truth and deconv results have different cell types"
    deconv_res = deconv_res.loc[gnd_truth.index,]
    
    # deconv_res is ctype by cells 
    
    # spearmann_cor = gnd_truth.corrwith(deconv_res, axis = 0, method='spearman') # if not doing rank correlation, normalization will matter 
    # avg_corr_sp = spearmann_cor.mean()

    pearson_cor = gnd_truth.corrwith(deconv_res, axis = 0, method='pearson') # if not doing rank correlation, normalization will matter 
    avg_corr_pe = pearson_cor.mean()
    
    # spearmann_cor_mat = gnd_truth.corrwith(deconv_res, axis = 1, method='spearman') # if not doing rank correlation, normalization will matter 
    # correlations_sp = pd.DataFrame(index=deconv_res.T.columns, columns=gnd_truth.T.columns)
    correlations_pe = pd.DataFrame(index=deconv_res.T.columns, columns=gnd_truth.T.columns)
    for col1 in deconv_res.T.columns:
        for col2 in gnd_truth.T.columns:
            # gives nan when all 0's due to cell type not present in ref or query
            # correlations_sp.at[col1, col2] = deconv_res.T[col1].corr(gnd_truth.T[col2], method='spearman')
            correlations_pe.at[col1, col2] = deconv_res.T[col1].corr(gnd_truth.T[col2], method='pearson')

    # print(correlations)
    
    gnd_truth_norm = np.array(gnd_truth/gnd_truth.sum())
    
    deconv_res = np.array(deconv_res/deconv_res.sum())
    
    # RMSE of each spot, averaged over spots
    orig_rmse = np.sqrt(np.mean((gnd_truth_norm - deconv_res)**2, axis=0)).mean() 
    # np.sqrt(((gnd_truth_norm - deconv_res) ** 2).sum())
    # add 0.000001 to denom to prevent divide by 0
    # new_rmse = np.sqrt(((((gnd_truth_norm - deconv_res) ** 2).sum(axis=0)/gnd_truth_norm.sum(axis=0)).sum())/gnd_truth_norm.shape[0])
    
    # ari = adjusted_rand_score(gt_dom_ct,pred_dom_ct)
    
    
    # contingency_matrix = metrics.cluster.contingency_matrix(gt_dom_ct,pred_dom_ct)
    # # return purity
    # purity =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    
    # code JSD
    #gnd_truth_norm_prop = gnd_truth_norm.div(gnd_truth_norm.sum(axis=1),axis='rows') # sum to 1 per-cell type across all spots 
    #deconv_res_prop = deconv_res.div(deconv_res.sum(axis=1),axis='rows') # sum to 1 per-cell type across all spots 
    #denom = (gnd_truth_norm_prop + deconv_res_prop)*0.5
    #gnd_truth_norm_prop = gnd_truth_norm/np.expand_dims(gnd_truth_norm.sum(axis=1),1)
    #deconv_res_prop = deconv_res/np.expand_dims(deconv_res.sum(axis=1),1)
    # denom = (gnd_truth_norm_prop + deconv_res_prop)*0.5
    denom = (gnd_truth_norm + deconv_res)*0.5
    # cell_type_jsd = 0.5*entropy(pk=deconv_res_prop, qk=denom,axis=1)  +  0.5*entropy(pk=gnd_truth_norm_prop, qk=denom,axis=1) # .shape #  base=base)
    cell_type_jsd = 0.5*entropy(pk=deconv_res, qk=denom,axis=0)  +  0.5*entropy(pk=gnd_truth_norm, qk=denom,axis=0) # .shape #  base=base)
    # only compute average for cell types that are in both reference and query 
    # cell_type_jsd = cell_type_jsd[~np.isnan(cell_type_jsd)]
    avg_jsd = cell_type_jsd.mean() # higher is worse because more deviation from actual and true distribution
    # perfectly correct should give 0
    # print([rmse, ari, purity,avg_corr], correlations)
    return [orig_rmse,   avg_corr_pe, avg_jsd], [correlations_pe,cell_type_jsd]      # , spear_correl


def eval_perspot(gnd_truth,deconv_res):
  
    gnd_truth = gnd_truth.T
    
    deconv_res.columns = gnd_truth.columns # keeping cell ids consistent 
    assert set(deconv_res.index) == set(gnd_truth.index), "the ground truth and deconv results have different cell types"
    deconv_res = deconv_res.loc[gnd_truth.index,]

    # pearson corr 
    pearson_cor = gnd_truth.corrwith(deconv_res, axis = 0, method='pearson') # if not doing rank correlation, normalization will matter 
    # avg_corr_pe = pearson_cor.mean()


    # dom ctype - True / False 
    pred_dom_ct = deconv_res.idxmax()
    gt_dom_ct = gnd_truth.idxmax()
    is_correct_dom = (pred_dom_ct == gt_dom_ct).astype(int)
    # is_correct_dom


    # correlation between cell-types 
    correlations_pe = pd.DataFrame(index=deconv_res.T.columns, columns=gnd_truth.T.columns)
    for col1 in deconv_res.T.columns:
        for col2 in gnd_truth.T.columns:
            # gives nan when all 0's due to cell type not present in ref or query
            # correlations_sp.at[col1, col2] = deconv_res.T[col1].corr(gnd_truth.T[col2], method='spearman')
            correlations_pe.at[col1, col2] = deconv_res.T[col1].corr(gnd_truth.T[col2], method='pearson')
    # correlations_pe



    # euclidean dist 
    gnd_truth_norm = np.array(gnd_truth/gnd_truth.sum())    
    deconv_res = np.array(deconv_res/deconv_res.sum())
    # orig_rmse = np.sqrt(((gnd_truth_norm - deconv_res) ** 2).sum())
    cell_rmse = np.sqrt(((gnd_truth_norm - deconv_res) ** 2).sum(axis=0))



    per_cell_metrics = pd.DataFrame(index=gnd_truth.columns,columns=['pearson_cor','correct_dom','cell_rmse'])

    per_cell_metrics['pearson_cor'] = pearson_cor
    per_cell_metrics['correct_dom'] = is_correct_dom
    per_cell_metrics['cell_rmse'] = cell_rmse

    return per_cell_metrics, correlations_pe
