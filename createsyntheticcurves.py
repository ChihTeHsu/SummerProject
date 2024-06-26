# -*- coding: utf-8 -*-
"""CreateSyntheticCurves.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TkZoxxr0vok_kRHSHv6TmgAE_eiPwxqY
"""

!pip install faiss-cpu
!pip install tensorflow_similarity==0.17.1

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

from scipy.special import erf
def IVcurve(x, Vx, Phi, Ti, Of):
    Vsc=7545
    pi=3.1415926
    q=1.6e-19
    k=1.38e-23
    mH=1.67262177774e-27
    mHe=(1.67262177774e-27)*4
    mO=(1.67262177774e-27)*16

    b_H= np.sqrt(mH/(2*k*Ti))
    b_He=np.sqrt(mHe/(2*k*Ti))
    b_O= np.sqrt(mO/(2*k*Ti))

    f_H= Vsc-Vx-np.sqrt( (2*q/mH)* (0.5+0.5*np.tanh(1e+19*(x+Phi))) * (x+Phi) )
    f_He=Vsc-Vx-np.sqrt( (2*q/mHe)*(0.5+0.5*np.tanh(1e+19*(x+Phi))) * (x+Phi) )
    f_O= Vsc-Vx-np.sqrt( (2*q/mO)* (0.5+0.5*np.tanh(1e+19*(x+Phi))) * (x+Phi) )

    IV= lambda b,f :0.5* (1 + erf(b*f) + np.exp(-b*b*f*f)/(np.sqrt(pi)*b*(Vsc-Vx) ))
    return (1-Of)*IV(b_H,f_H)+ Of*IV(b_O,f_O)

RV = np.array(
    [-9.93373125e-04,3.95764436e-01,6.93589837e-01,8.93694317e-01,1.09104005e+00,1.18945533e+00,1.28752994e+00,1.38576693e+00,
     1.58509721e+00,1.78320376e+00,2.17832343e+00,2.47770206e+00,2.87403719e+00,3.17350849e+00,3.57345597e+00,3.87276497e+00,
     4.17414676e+00,4.47308281e+00,4.77574920e+00,4.97875069e+00,5.18096931e+00,5.38201325e+00,5.78615054e+00,6.28558492e+00,
     6.88648143e+00,7.48722301e+00,7.98688117e+00,8.48436093e+00,9.97502566e+00,1.05e+01,1.1e+01,1.2e+01])

ground_pars = []
DATA = []
for i in range(30000):
    Vx = np.random.uniform(-300, 300)
    Phi = -0.1
    Ti = np.random.uniform(1000, 3000)
    Of = np.random.uniform(0, 1)
    #noise = np.random.normal(0, 0.01, RV.shape[-1])
    ground_pars.append([Vx, Ti, Of])
    DATA.append( IVcurve(RV, Vx, Phi, Ti, Of) )
ground_pars = np.array(ground_pars)
DATA = np.array(DATA)

from sklearn.preprocessing import KBinsDiscretizer
enc = KBinsDiscretizer(n_bins=7, encode="ordinal", strategy='kmeans')
LABELS = enc.fit_transform(ground_pars)

from sklearn.model_selection import train_test_split

X_train, X_ref, y_train, y_ref = train_test_split(DATA, LABELS, test_size=0.1, random_state=42)
X_ref, X_test, y_ref, y_test = train_test_split(X_ref, y_ref, test_size=0.02, random_state=42)

print(X_train.shape)
print(X_ref.shape)
print(X_test.shape)

import faiss

index = faiss.IndexFlatL2(X_ref.shape[1])
index.add(X_ref)

# predict the test
distances, indices = index.search(X_test, k=4)

distances[:3]

indices[:3]

votes = y_ref[indices[:3]]
votes

from scipy import stats

mode, _ = stats.mode(votes[:,:,0],axis=-1)
mode

mode

y_test[:3, 0]

y_test[:3, 0] == mode

import faiss
from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.utils import unpack_results
from collections.abc import MutableMapping

class myCallback(EvalCallback):
    def __init__(self, queries, query_labels, query_values, targets, target_labels, target_values, k):
        super().__init__(queries, query_labels, targets, target_labels, k)
        self.query_values = query_values
        self.target_values = target_values

    def on_epoch_end(self, epoch: int, logs: MutableMapping | None = None):
        _ = epoch
        if logs is None:
            logs = {}

        targets_emb = model.predict(self.targets)
        queries_emb = model.predict(self.queries_known)

        # rebuild the index
        index = faiss.IndexFlatL2(targets_emb.shape[1])
        index.add(targets_emb)

        # predict the queries
        distances, indices = index.search(queries_emb, k=self.k)
        votes = self.target_values[indices]

        #tf.print( self.query_values.shape )
        res0 = np.abs(self.query_values[:,0] - np.mean(votes[:,:,0],axis=-1)) # Vx
        res1 = np.abs(self.query_values[:,1] - np.mean(votes[:,:,1],axis=-1)) # AP
        res2 = np.abs(self.query_values[:,2] - np.mean(votes[:,:,2],axis=-1)) # Ti
        res3 = np.abs(self.query_values[:,3] - np.mean(votes[:,:,3],axis=-1)) # Hf
        #res = np.sum(np.abs(self.query_values[:,:] - votes[:,0,:]), axis=-1)
        #res = np.sum(np.abs((self.query_values[:,:] - votes[:,0,:])/self.query_values[:,:]), axis=-1)
        known_results = {"Vx": res0}, {"AP": res1}, {"Ti": res2}, {"Hf": res3}

        for a in known_results:
            unpack_results(
                a,
                epoch=epoch,
                logs=logs,
                tb_writer=self.tb_writer,
            )
        #self.model.reset_index()