import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

header = ['user_id','item_id','rating','timestamp']

df = pd.read_csv('u.data',sep=',',names=header,engine='python',header=None)

#print df['item_id'].unique()

n_users = df.user_id.unique().shape[0]

n_items = df.item_id.unique().shape[0]

from sklearn import cross_validation as cv

train_data, test_data = cv.train_test_split(df,test_size=0.25)

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

u, s, vt = svds(train_data_matrix,k = 20)

#print train_data_matrix.shape

s_diag_matrix = np.diag(s)

i = 15

u = u[:,:i]

print u.shape

s_diag_matrix = s_diag_matrix[:i,:i]

print s_diag_matrix.shape

vt = vt[:i,:]


X_pred = np.dot(np.dot(u, s_diag_matrix),vt)

test_user = 64

user_recomm = X_pred[test_user-1,]

#print user_recomm.shape

#print max(list(enumerate(user_recomm)),key=(lambda x:x[1]))

tmp = list(enumerate(user_recomm))

tmp.sort(key=lambda x: int(x[1]),reverse=True)

t10 = tmp[:10]

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movie = pd.read_csv('u.item',sep='|',names=i_cols,encoding='latin-1')

#print movie['movie title']

for id in t10:
    print movie.loc[movie['movie id'] == id[0]+1]['movie title']

print 'User-based CF MSE: ' + str(rmse(X_pred,test_data_matrix))
