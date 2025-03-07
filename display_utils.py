import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
def k_PCA(X, k):
    pca = PCA()
    X = pca.fit_transform(X)
    return X[:,:k]

import matplotlib.lines as mlines
def display_pca_and_labels(X, y, compresion_mask=[]):
    X = k_PCA(X, k=2)
    pc1 = X[:,0]
    pc2 = X[:,1]
    
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    y_colors = cmap(norm(y))
    
    if compresion_mask is None:
        plt.scatter(pc1, pc2, facecolors='none', c=y, cmap='coolwarm')
    else:
        kept_points = compresion_mask
        removed_points = ~compresion_mask
        plt.scatter(pc1[kept_points], pc2[kept_points], c=y_colors[kept_points], cmap='coolwarm')
        plt.scatter(pc1[removed_points], pc2[removed_points], facecolors='none', edgecolors=y_colors[removed_points], cmap='coolwarm')
     
    filled_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label="Point gardés")
    if compresion_mask is None:
        handles_list = [filled_marker]
    else:
        hollow_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, markerfacecolor='none', label="Points enlevés")
        handles_list = [filled_marker, hollow_marker]
    

    plt.legend(handles=handles_list, loc="upper right", title="Légende")

        
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - 2 Composantes principales de X avec valeurs y en couleur")
    plt.grid()
    plt.show()  
    
def display_only_kept_points(X, y, compresion_mask):
    X = k_PCA(X, k=2)
    pc1 = X[:,0]
    pc2 = X[:,1]
    
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    y_colors = cmap(norm(y))
    
    kept_points = compresion_mask
    plt.scatter(pc1[kept_points], pc2[kept_points], c=y_colors[kept_points], cmap='coolwarm')
     
    filled_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label="Point gardés")

    plt.legend(handles=[filled_marker], loc="upper right", title="Légende")

        
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - 2 Composantes principales de X avec valeurs y en couleur")
    plt.grid()
    plt.show()  
    
    