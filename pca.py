import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv('pca.csv', index_col='name')

_ = plt.scatter(df['x_value'], df['y_value'])
_ = plt.xlabel('Score chocolate')
_ = plt.ylabel('Score candy')
_ = plt.title('Score of chocolate & candy')
_ = plt.axis('equal')
_ = plt.annotate('',  xytext=(1, 1), xy=(50, 20), arrowprops=dict(arrowstyle="->", color='black'))

# Eigen of Covaricen data
# (Covariance = auto centered)
raw_eig_val, raw_eig_vec = np.linalg.eig(df.cov())
raw_eig_val
raw_eig_vec.round(3)

# Eigen of centered orthogonalized data
# no need to use cov matrix
# since double centered data again in cov()

# Centered data
cen_df = df - df.mean()
mat = cen_df.values

# Orthogonalize matrix with
# M transpose x M
orth_mat = mat.T.dot(mat)
cen_eig_val, cen_eig_vec = np.linalg.eig(orth_mat)
cen_eig_val
cen_eig_vec.round(3)

# Eigen of Correlation
# all the eigvalue , vector differ
cor_df = df.corr()
cor_eig_val, cor_eig_vec = np.linalg.eig(cor_df)
cor_eig_val
cor_eig_vec.round(3)

# SVD of centered, orthogonalized data
# = eigen of centerd, orthognalized data
cen_df = df - df.mean()
mat = cen_df.values
orth_mat = mat.T.dot(mat)

U, S, V = np.linalg.svd(cen_df)
# S**2 = eigen of center, orthogonal data
# S = non orthgonal
S
# principal component , differ in sign
V.T.round(3)

# Visualized PCA
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='<-', linewidth=2, shrinkA=0, shrinkB=0, color='black')
    ax.annotate('', v0, v1, arrowprops=arrowprops)

_ = plt.scatter(df['x_value'], df['y_value'], alpha=0.5)

for length, vector in zip(cen_eig_val, cen_eig_vec):
    v = vector * 3 * np.sqrt(length)
    draw_vector(df.mean(), df.mean() + v)

def plot_vector_linear_transform(X, A=None):
    """
    X (vector)
    A (Metrix of linear transform)
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    if A is None:
        ax.quiver(X[0], X[1], color='gray', units='xy', scale=1)

    else:
        Z = A.dot(X)
        ax.quiver(X[0], X[1], color='gray', units='xy', scale=1)
        ax.quiver(Z[0], Z[1], color='blue', units='xy', scale=1)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

X = np.array([2,0])
A = np.array([[0,1],[-1,0]]).T

plot_vector_linear_transform(X, A)

# Eigen vector test
A = np.array([[2,1],[2,3]]).T
eig_val, eig_vec = np.linalg.eig(A)
eig_val
eig_vec
n = len(eig_val)
for i in range(0, n):
    print('Eigen val {}'.format(eig_val[i]))
    print('Corresponsing Eigen Vector')
    print(eig_vec[:,i])
    print('Result from transformation A.dot(eigen_vector)')
    print(A.dot(eig_vec[:,i]))
    print('Result from scaling eigen_vector*eigen_val')
    print(eig_vec[:,i]*eig_val[i])

# Q analysis on lipset data
df = pd.read_csv(os.path.join(os.getcwd(), 'lipset.csv'))
df.head()
df.shape

cor_mat = df.corr()
U,S,V = np.linalg.svd(cor_mat)
S[1]
V[1]
cor_mat.dot(V[1])/S[1]
U.shape
S.shape
V.shape

# Find cutting
# with diminishing varience explained
exp = (S/S.sum()).cumsum()
fix, ax = plt.subplots()
plt.plot(exp)
plt.plot(np.diff(exp))
ax.set_aspect('equal')
