import numpy as np

def neighborhood_analysis(samples: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    '''
    we assume samples only has numbers in it, so each row only contains the expression value for each gene
    
    samples.shape = S, G
        S -> Number of samples (or in modern terms, batch dimension.)
        G -> Number of genes used as probes in DNA Microarray
        
    class_labels.shape = S
    
    returns a new samples array with only genes that have meaningful correlation with the output classes
    '''
    
    if class_labels.ndim > 1:
        class_labels = np.reshape(class_labels, (-1,))
    
    classes = set(list(class_labels))
    num_genes = samples.shape[-1]
    best_genes_set = set()
    
    for cl in classes:
        ideal_exp = np.where(class_labels == cl, 1, -1).astype(int)
        random_exp = ideal_exp.copy()
        np.random.shuffle(random_exp)
        sim_and_gene_ideal = []
        max_random_sim = 0
        for gene_idx in range(num_genes):
            gene_vec = samples[..., gene_idx]
            sim_ideal = np.corrcoef(gene_vec, ideal_exp)[0][1]
            sim_random = np.corrcoef(gene_vec, random_exp)[0][1]
            sim_and_gene_ideal.append((sim_ideal, gene_idx))
            max_random_sim = max(max_random_sim, sim_random)
        filtered_genes = [gene_idx for sim, gene_idx in sim_and_gene_ideal if sim > max_random_sim]
        for gene_idx in filtered_genes:
            best_genes_set.add(gene_idx)
    
    new_samples = []
    for gene_idx in best_genes_set:
        gene_vec = samples[..., gene_idx]
        new_samples.append(gene_vec)
    new_samples = np.array(new_samples).T
    
    return new_samples