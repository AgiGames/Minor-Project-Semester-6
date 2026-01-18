import numpy as np
from sklearn.preprocessing import StandardScaler

class Predictor:
    
    def __init__(self, samples: np.ndarray, class_labels: np.ndarray):
        self.samples = samples
        self.class_labels = class_labels
        
    def neighborhood_analysis(self) -> np.ndarray:
        '''
        we assume samples only has numbers in it, so each row only contains the expression value for each gene
        
        samples.shape = S, G
            S -> Number of samples (or in modern terms, batch dimension.)
            G -> Number of genes used as probes in DNA Microarray
            
        self.class_labels.shape = S
        
        returns a new samples array with only genes that have meaningful correlation with the output classes
        '''
        
        if self.class_labels.ndim > 1:
            self.class_labels = np.reshape(self.class_labels, (-1,))
        
        classes = set(list(self.class_labels))
        num_genes = self.samples.shape[-1]
        best_genes_set = set()
        
        for cl in classes:
            ideal_exp = np.where(self.class_labels == cl, 1, -1).astype(int)
            random_exp = ideal_exp.copy()
            np.random.shuffle(random_exp)
            sim_and_gene_ideal = []
            max_random_sim = 0
            for gene_idx in range(num_genes):
                gene_vec = self.samples[..., gene_idx]
                sim_ideal = np.corrcoef(gene_vec, ideal_exp)[0][1]
                sim_random = np.corrcoef(gene_vec, random_exp)[0][1]
                sim_and_gene_ideal.append((sim_ideal, gene_idx))
                max_random_sim = max(max_random_sim, sim_random)
            filtered_genes = [gene_idx for sim, gene_idx in sim_and_gene_ideal if sim > max_random_sim]
            for gene_idx in filtered_genes:
                best_genes_set.add(gene_idx)
        
        best_genes = list(best_genes_set)
        new_samples = []
        for gene_idx in best_genes:
            gene_vec = self.samples[..., gene_idx]
            new_samples.append(gene_vec)
        new_samples = np.array(new_samples).T
        
        self.samples = new_samples
        return best_genes
        
    def calc_weights(self):
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
        
        classes = np.unique(self.class_labels)
        num_genes = self.samples.shape[-1]
        weights = []
        
        for cl in classes:
            ideal_exp = np.where(self.class_labels == cl, 1, -1).astype(int)
            cl_weights = []
            for gene_idx in range(num_genes):
                gene_vec = self.samples[..., gene_idx]
                sim_ideal = np.corrcoef(gene_vec, ideal_exp)[0][1]
                cl_weights.append(sim_ideal)
            cl_weights = np.array(cl_weights)
            weights.append(cl_weights)
        
        self.weights = np.array(weights)
        self.weights_classes = classes
        
    def fit(self):
        self.calc_weights()
    
    def predict(self, x_samples: np.ndarray):
        ps = x_samples @ self.weights.T
        class_idx = np.argmax(ps, axis=1)
        return self.weights_classes[class_idx]