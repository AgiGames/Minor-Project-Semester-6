import numpy as np

def fit_predictor(samples: np.ndarray, class_labels: np.ndarray):
    '''
    we assume samples only has numbers in it, so each row only contains the expression value for each gene
    
    samples.shape = S, G
        S -> Number of samples (or in modern terms, batch dimension.)
        G -> Number of genes used as probes in DNA Microarray
        
    class_labels.shape = S
    
    returns weights for each class
    weights.shape = C, G
        C -> Number of classes
        G -> Number of genes
    '''
    
    if class_labels.ndim > 1:
        class_labels = np.reshape(class_labels, (-1,))
    
    classes = np.unique(class_labels)
    num_genes = samples.shape[-1]
    weights = []
    
    for cl in classes:
        ideal_exp = np.where(class_labels == cl, 1, -1).astype(int)
        cl_weights = []
        for gene_idx in range(num_genes):
            gene_vec = samples[..., gene_idx]
            sim_ideal = np.corrcoef(gene_vec, ideal_exp)[0][1]
            cl_weights.append(sim_ideal)
        cl_weights = np.array(cl_weights)
        weights.append(cl_weights)
        
    return np.array(weights), classes
        