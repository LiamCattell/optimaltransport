import numpy as np

n_samp = 5
n_feat = 3
X = np.random.rand(n_samp,n_feat)

# Row-wise multiplication
mat1 = np.zeros((n_feat,n_feat))
for row in X:
    row = row.reshape(n_feat,1)
    mat1 += row.dot(row.T)
print(mat1)

# Full matrix multiplication
mat2 = X.T.dot(X)
print(mat2)
