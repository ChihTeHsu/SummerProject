# -*- coding: utf-8 -*-
"""RealCurvesSimilarity.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1luWlszep4Bo5D8YRYJ7NnyN3RSbav5bR
"""

!pip install tensorflow_similarity==0.17.1
!pip install tensorflow_addons
!pip install faiss-cpu

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive/')
# %cd /content/drive/My Drive/

import os
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(seed=777)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

TRAIN = np.load('/content/drive/MyDrive/Colab Notebooks/SpaceScience/Dataset/train_IVs&pars.npy', allow_pickle=True).item()
RI = TRAIN['ri'].reshape(-1,32)
RI = RI/np.mean(RI[:,:4],axis=-1,keepdims=True)
RV = TRAIN['rv'].reshape(-1,32)

idx, _ = np.where( np.isnan(RI))
RI = np.delete(RI, idx, axis=0)
RV = np.delete(RV, idx, axis=0)

VAL = np.load('/content/drive/MyDrive/Colab Notebooks/SpaceScience/Dataset/dic_IVs&pars.npy', allow_pickle=True).item()
ri = VAL['ri'].reshape(-1,32)
ri = ri/np.mean(ri[:,:4],axis=-1,keepdims=True)
rv = VAL['rv'].reshape(-1,32)
vst = VAL['ground_pars']['vst']
vx = VAL['ground_pars']['vx']
v_cor_x = VAL['ground_pars']['v_cor_x']
Ti = VAL['ground_pars']['Ti']
O_frac = VAL['ground_pars']['O_frac']
H_frac = VAL['ground_pars']['H_frac']
AP_POT = VAL['ground_pars']['AP_POT']

idx = np.where(O_frac>0.8)[0]
ri = np.delete(ri, idx, axis=0)
rv = np.delete(rv, idx, axis=0)
vst = np.delete(vst, idx, axis=0)
vx = np.delete(vx, idx, axis=0)
v_cor_x = np.delete(v_cor_x, idx, axis=0)
Ti = np.delete(Ti, idx, axis=0)
O_frac = np.delete(O_frac, idx, axis=0)
H_frac = np.delete(H_frac, idx, axis=0)
AP_POT = np.delete(AP_POT, idx, axis=0)

LABELS = KBinsDiscretizer(
    n_bins=12,
    encode="ordinal",
    strategy='kmeans',
    random_state=777
    ).fit_transform(np.c_[0.5*(1.67262177774e-27/1.6e-19)*(16)*(vst - vx + v_cor_x)**2 + (1)*AP_POT, Ti, O_frac])

RI_ref, RI_test, RV_ref, RV_test, y_ref, y_test = train_test_split(ri, rv, LABELS, test_size=0.1, random_state=26)

import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 26
BATCH_SIZE = 128
INTERP_TO = 128
PROB = 0.1
XGRID = np.linspace(0, 12, INTERP_TO)

import typing
def tf_interp(x: typing.Any, xs: typing.Any, ys: typing.Any) -> tf.Tensor:
    ys = tf.convert_to_tensor(ys)
    dtype = ys.dtype

    ys = tf.cast(ys, tf.float64)
    xs = tf.cast(xs, tf.float64)
    x = tf.cast(x, tf.float64)

    xs = tf.concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    bs = ys - ms*xs

    i = tf.math.argmax(xs[..., tf.newaxis, :] > x[..., tf.newaxis], axis=-1)
    m = tf.gather(ms, i, axis=-1)
    b = tf.gather(bs, i, axis=-1)

    y = m*x + b
    return tf.cast(tf.reshape(y, tf.shape(x)), dtype)

RNG = tf.random.Generator.from_seed(33)
def custom_augment(rv, ri):
    return tf.nn.experimental.general_dropout(tf_interp(XGRID, rv, ri + RNG.normal((ri.shape[-1],), 0, 0.015, dtype=ri.dtype)), PROB, RNG.uniform)*(1-PROB)

train_one = (
    tf.data.Dataset.from_tensor_slices( (RV, RI) ).shuffle(buffer_size=4096, seed=SEED)
    .map(custom_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
train_two = (
    tf.data.Dataset.from_tensor_slices( (RV, RI) ).shuffle(buffer_size=4096, seed=SEED)
    .map(custom_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
train_ds = tf.data.Dataset.zip((train_one, train_two))

ref_ds = tf.data.Dataset.from_tensor_slices( (RV_ref, RI_ref) ).map(lambda rv, ri: tf_interp(XGRID, rv, ri), num_parallel_calls=tf.data.AUTOTUNE).batch(9)
test_ds = tf.data.Dataset.from_tensor_slices( (RV_test, RI_test) ).map(lambda rv, ri: tf_interp(XGRID, rv, ri), num_parallel_calls=tf.data.AUTOTUNE).batch(9)

import faiss
from scipy import stats
from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.utils import unpack_results
from collections.abc import MutableMapping

class myCallback(EvalCallback):
    def __init__(self, queries, query_labels, targets, target_labels, k):
        super().__init__(queries, query_labels, targets, target_labels)
        """
        Args:
            queries: Test examples that will be tested against the built index.
            query_labels: Test examples expected ground truth labels.
            targets: Reference examples that are used to build index.
            target_labels: Reference examples labels.
            k: Number of neighbors to return for each query.
        """
        self.target_labels = target_labels # initial as numpy for fancy indexing later
        self.k = k

    def on_epoch_end(self, epoch: int, logs: MutableMapping | None = None):
        _ = epoch
        if logs is None:
            logs = {}

        # get the embeddings
        reference = self.model.backbone.predict(self.targets, verbose=0)
        test = self.model.backbone.predict(self.queries_known, verbose=0)

        # rebuild the index from reference
        index = faiss.IndexFlatL2(reference.shape[1])
        index.add(reference)

        # predict the test
        distances, indices = index.search(test, k=self.k)

        # lookup reference label via predicted indices
        votes = self.target_labels[indices] # unless self.target_labels is initial as numpy, otherwise we cannot use fancy indexing here
        #votes = tf.stack([self.target_labels.numpy()[indices[:,k]] for k in range(self.k)], axis=1) # cannot directly use fancy indexing style in the tensorflow

        # take mode as prediction output
        Vxmode, _ = stats.mode(votes[:,:,0],axis=-1,keepdims=False)
        Timode, _ = stats.mode(votes[:,:,1],axis=-1,keepdims=False)
        Ofmode, _ = stats.mode(votes[:,:,2],axis=-1,keepdims=False)

        # count how many times it correct
        VxRes = np.sum(self.query_labels_known[:, 0] == Vxmode)
        TiRes = np.sum(self.query_labels_known[:, 1] == Timode)
        OfRes = np.sum(self.query_labels_known[:, 2] == Ofmode)

        known_results = {"Vx": VxRes}, {"Ti": TiRes}, {"Of": OfRes}
        for a in known_results:
            unpack_results(
                a,
                epoch=epoch,
                logs=logs,
                tb_writer=self.tb_writer,
            )
        self.model.reset_index()

import faiss

reference = np.concatenate([x for x in tfds.as_numpy(ref_ds)])
test = np.concatenate([x for x in tfds.as_numpy(test_ds)])

# rebuild the index from reference
index = faiss.IndexFlatL2(reference.shape[1])
index.add(reference)

# predict the test
distances, indices = index.search(test, k=4)

# lookup reference label via predicted indices
votes = y_ref[indices] # unless self.target_labels is initial as numpy, otherwise we cannot use fancy indexing here
#votes = tf.stack([self.target_labels.numpy()[indices[:,k]] for k in range(self.k)], axis=1) # cannot directly use fancy indexing style in the tensorflow

# take mode as prediction output
Vxmode, _ = stats.mode(votes[:,:,0],axis=-1,keepdims=False)
Timode, _ = stats.mode(votes[:,:,1],axis=-1,keepdims=False)
Ofmode, _ = stats.mode(votes[:,:,2],axis=-1,keepdims=False)

# count how many times it correct
VxRes = np.sum(y_test[:, 0] == Vxmode)
TiRes = np.sum(y_test[:, 1] == Timode)
OfRes = np.sum(y_test[:, 2] == Ofmode)

print(VxRes, TiRes, OfRes)

import tensorflow_addons as tfa
import tensorflow_similarity as tfsim

TEMPERATURE = 0.25
INIT_LR = 1e-3
EPOCHS = 12
HID_SIZE = 16

def get_backbone(input_dim, channel=16):
    inputs = tf.keras.layers.Input((input_dim,), name="backbone_input")
    x = tf.expand_dims(inputs, axis=-1)
    x = tf.keras.layers.Conv1D(channel, kernel_size=5, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv1D(channel, kernel_size=5, strides=2, activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    #x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(channel, activation="relu", name="backbone_output")(x)
    return tf.keras.Model(inputs, outputs, name="backbone")

backbone = get_backbone( INTERP_TO, channel=32)
projector = tfsim.models.contrastive_model.get_projector( input_dim = backbone.output.shape[-1], dim=HID_SIZE, num_layers=2)

model = tfsim.models.ContrastiveModel(
    backbone = backbone,
    projector = projector,
    algorithm="simclr"
)

model.compile(
    optimizer = tfa.optimizers.LAMB(learning_rate=INIT_LR),
    loss = tfsim.losses.SimCLRLoss(name="simclr", temperature=TEMPERATURE)
)

history = model.fit(
    train_ds,
    epochs = EPOCHS,
    callbacks=[
        myCallback(test_ds, y_test, ref_ds, y_ref, k=4)
    ],
    verbose = 1,
)

import matplotlib.pyplot as plt
plt.figure(figsize=(17,4))

ax1 = plt.subplot(141)
plt.plot(history.history["loss"])
plt.grid()
plt.title("simclr_loss")

ax2 = plt.subplot(142)
plt.plot(history.history["Vx"], label="error")
plt.grid()
plt.title("Vx")

ax3 = plt.subplot(143)
plt.plot(history.history["Ti"], label="error")
plt.grid()
plt.title("Ti")

ax4 = plt.subplot(144)
plt.plot(history.history["Of"], label="error")
plt.grid()
plt.title("Of")

plt.show()