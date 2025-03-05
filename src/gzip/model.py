from tqdm import trange
import gzip
from sklearn.neighbors import KNeighborsClassifier
import pickle

class GZIP:
    def __init__(self):
        pass


    def load(self):

        
        train_x, train_y = train['text'], train['label']
        train_x, train_y = train_x[0:2000], train_y[0:2000]
        test_x, test_y = test['text'],test['label']
        test_x, test_y = test_x[0:100], test_y[0:100]
        def ncd(x, x2): # NCD with compressed lengths
            x_compressed = len(gzip.compress(x.encode()))
            x2_compressed = len(gzip.compress(x2.encode()))  
            xx2 = len(gzip.compress((" ".join([x,x2])).encode()))
            return (xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed)

        train_ncd = [[ncd(train_x[i], train_x[j]) for j in range(len(train_x))] for i in trange(len(train_x))]
        test_ncd = [[ncd(test_x[i], train_x[j]) for j in range(len(train_x))] for i in trange(len(test_x))]

        # KNN classification
        neigh = KNeighborsClassifier(n_neighbors=4,leaf_size=20) 
        neigh.fit(train_ncd, train_y)
        print("Accuracy:", neigh.score(test_ncd, test_y))