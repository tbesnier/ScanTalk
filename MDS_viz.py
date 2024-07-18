import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import MDS
import numpy as np



# Load data

X_VOCA_original = np.load("/home/tbesnier/phd/projects/Data/scantalk_extension/latent_trajectories/VOCA_original.npy")[:60]
y_VOCA_original = np.zeros(X_VOCA_original.shape[0])[:60]
X_VOCA_remeshed = np.load("/home/tbesnier/phd/projects/Data/scantalk_extension/latent_trajectories/VOCA_remeshed.npy")[:60]
y_VOCA_remeshed = np.ones(X_VOCA_remeshed.shape[0])[:60]
X_arnold = np.load("/home/tbesnier/phd/projects/Data/scantalk_extension/latent_trajectories/arnold.npy")[:60]
y_arnold = 2*np.ones(X_arnold.shape[0])[:60]
target_names = ["VOCA original", "VOCA remeshed", "Arnold"]

X = np.vstack((X_VOCA_original, X_VOCA_remeshed, X_arnold))
y = np.concatenate((y_VOCA_original, y_VOCA_remeshed, y_arnold))

# Apply MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))

# Scatter plot for each class
colors = ['navy', 'turquoise', 'darkorange']
start_color = 'red'
end_color = 'green'
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    class_indices = (y == i)
    X_class = X_mds[class_indices]
    plt.scatter(X_mds[y == i, 0], X_mds[y == i, 1], color=color, label=target_name, edgecolor='k', s=50)
    # Line plot to connect points
    plt.plot(X_mds[y == i, 0], X_mds[y == i, 1], color=color, linestyle='-', linewidth=1)

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS')
plt.legend()
plt.show()
