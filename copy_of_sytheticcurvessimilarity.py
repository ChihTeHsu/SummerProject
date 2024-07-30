# -*- coding: utf-8 -*-
"""Copy of SytheticCurvesSimilarity 的副本

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KCUf4HwWajxWkqVyR_lIuFORRTbBJMS9
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

data = np.load('/content/drive/MyDrive/Colab Notebooks/SpaceScience/Dataset/train_IVs&pars_uniform.npy', allow_pickle=True).item()

ri = data['ri'].reshape(-1,32)
ri = ri/np.mean(ri[:,:4],axis=-1,keepdims=True)
rv = data['rv'].reshape(-1,32)
vst = data['ground_pars']['vst']
vx = data['ground_pars']['vx']
v_cor_x = data['ground_pars']['v_cor_x']
Ti = data['ground_pars']['Ti']
O_frac = data['ground_pars']['O_frac']
H_frac = data['ground_pars']['H_frac']
AP_POT = data['ground_pars']['AP_POT']

idx, _ = np.where(np.isnan(ri))
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
    n_bins=30,
    encode="ordinal",
    strategy='kmeans',
    random_state=777
    ).fit_transform(np.c_[0.5*(1.67262177774e-27/1.6e-19)*(16)*(vst - vx + v_cor_x)**2 + (1)*AP_POT, Ti, O_frac])

RI_ref, RI_test, RV_ref, RV_test, y_ref, y_test = train_test_split(ri, rv, LABELS, test_size=0.1, random_state=777)

from scipy.special import erf
def IVcurve(x, Vx, Phi, Ti, Of):
    Vsc=7114
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
for i in range(300000):
    Vx = np.random.uniform(-1000, 1000)
    Phi = np.random.uniform(-2.5, 0.)
    Ti = np.random.uniform(200, 4500)
    Of = np.random.uniform(0, 1)
    ground_pars.append([Vx, Ti, Of])
    DATA.append( IVcurve(RV, Vx, Phi, Ti, Of) )
ground_pars = np.array(ground_pars)
DATA = np.array(DATA)

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
        reference = self.model.predict(self.targets, verbose=0)
        test = self.model.predict(self.queries_known, verbose=0)

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

class FocalModulation(tf.keras.layers.Layer):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop_rate=0., use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = tf.keras.layers.Dense(2*dim + (self.focal_level+1), use_bias=bias, name="f" ) # nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = tf.keras.layers.Conv1D(filters=dim, kernel_size=1, strides=1, use_bias=bias, name="h" )

        self.act = tf.keras.layers.Activation("gelu") #nn.GELU()
        self.proj = tf.keras.layers.Dense(dim, name="proj" ) #nn.Linear(dim, dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop_rate)
        self.gap = tf.keras.layers.GlobalAveragePooling1D(  keepdims=True)
        self.focal_layers = []

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                tf.keras.Sequential([
                        #ZeroPadding2D(padding=(kernel_size // 2, kernel_size // 2)),
                        tf.keras.layers.Conv1D(filters=dim, kernel_size=kernel_size, strides=1,
                        groups=dim, padding='same', use_bias=False ,
                        activation=tf.keras.activations.gelu, dtype=tf.keras.backend.floatx()
                        )]
                        )
                )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-5  )

    def call(self, x, return_modulator=False):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x) #tf.transpose(self.f(x), perm=[0,3,1,2] ) #.permute(0, 3, 1, 2)#.contiguous()
        q, ctx, self.gates = tf.split(x, [self.dim, self.dim, self.focal_level+1], axis=-1)

        # context aggreation
        ctx_all = 0


        for l in range(self.focal_level):
            ctx = tf.cast(self.focal_layers[l](ctx), dtype=ctx.dtype)
            ctx_all = ctx_all + ctx*self.gates[..., l:l+1]
        ctx_global = self.act(  self.gap(ctx) ) #.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[..., self.focal_level:]
        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        #x_out = tf.transpose(x_out, perm=[0, 2, 3, 1]    ) #.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        if return_modulator:
            return x_out, self.modulator
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    Args:
        q: query shape == [..., seq_len_q, depth]
        k: key shape == [..., seq_len_k, depth]
        v: value shape == [..., seq_len_v, depth_v]
        mask: Float tensor with shape broadcastable to [..., seq_len_q, seq_len_k].

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # [..., seq_len_q, seq_len_k]
    output = tf.matmul(attention_weights, v)  # [..., seq_len_q, depth_v]
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(output_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into [num_heads, depth].
        Transpose the result such that the shape is [batch_size, num_heads, seq_len, depth]
        """
        # [batch_size, seq_len, num_heads, depth]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]  # q.shape: [batch_size, seq_len, d_model]

        q = self.wq(q)  # [batch_size, seq_len, d_model]
        k = self.wk(k)  # [batch_size, seq_len, d_model]
        v = self.wv(v)  # [batch_size, seq_len, d_model]

        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, seq_len_k, depth]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, seq_len_v, depth]

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # scaled_attention.shape == [batch_size, num_heads, seq_len_q, depth]
        # attention_weights.shape == [batch_size, num_heads, seq_len_q, seq_len_k]

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # [batch_size, seq_len_q, num_heads, depth]
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model))
        # [batch_size, seq_len_q, d_model]

        output = self.dense(concat_attention)  # [batch_size, seq_len_q, output_dim]
        return output

class EncBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, d_model, num_heads, ff_dim, rate=0.):
        super(EncBlock, self).__init__()
        self.att = MultiHeadAttention(d_model=d_model, num_heads=num_heads, output_dim=embed_dim)
        #self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="leaky_relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#x = EncBlock(embed_dim=3, d_model=16, num_heads=4, ff_dim=32, rate=0.1)(x)

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_similarity as tfsim
import tensorflow_datasets as tfds

SEED = 777
BATCH_SIZE = 1024
INTERP_TO = 64
#PROB = 0.3
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

# https://stackoverflow.com/questions/58546373/how-to-add-randomness-in-each-iteration-of-tensorflow-dataset
# https://stackoverflow.com/questions/69108284/tf-data-dataset-map-functionality-and-random
# https://github.com/tensorflow/tensorflow/issues/35090#issuecomment-1345470684
#def custom_augment(ri):
#    return tf.nn.dropout(tf_interp(XGRID, RV, ri + tf.random.normal((ri.shape[-1],), 0, 0.015, dtype=ri.dtype)), rate=PROB)*(1-PROB)
RNG = tf.random.Generator.from_seed(33)
def custom_augment(ri):
    noise1 = tf.nn.experimental.general_dropout(RNG.normal(ri.shape, 0, 0.02, dtype=ri.dtype), 0.9, RNG.uniform)*(1-0.9)
    noise2 = tf.nn.experimental.general_dropout(RNG.normal(ri.shape, 0, 0.03, dtype=ri.dtype), 0.7, RNG.uniform)*(1-0.7)
    return tf_interp(XGRID, RV, ri+noise1+noise2) + RNG.normal(XGRID.shape, 0, 0.01, dtype=ri.dtype)
    #return tf_interp(XGRID, RV, ri)+RNG.normal(XGRID.shape, 0, 0.01, dtype=ri.dtype)+tf.nn.experimental.general_dropout(RNG.normal(XGRID.shape, 0, 0.04, dtype=ri.dtype), PROB, RNG.uniform)*(1-PROB)
    #return tf.nn.experimental.general_dropout(tf_interp(XGRID, RV, ri + RNG.normal((ri.shape[-1],), 0, 0.025, dtype=ri.dtype)), PROB, RNG.uniform)*(1-PROB)

train_one = (
    tf.data.Dataset.from_tensor_slices( DATA ).shuffle(buffer_size=4096, seed=SEED)
    .map(custom_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
train_two = (
    tf.data.Dataset.from_tensor_slices( DATA ).shuffle(buffer_size=4096, seed=SEED)
    .map(custom_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
train_ds = tf.data.Dataset.zip((train_one, train_two))

ref_ds = tf.data.Dataset.from_tensor_slices( (RV_ref, RI_ref) ).map(lambda rv, ri: tf_interp(XGRID, rv, ri), num_parallel_calls=tf.data.AUTOTUNE).batch(9)
test_ds = tf.data.Dataset.from_tensor_slices( (RV_test, RI_test) ).map(lambda rv, ri: tf_interp(XGRID, rv, ri), num_parallel_calls=tf.data.AUTOTUNE).batch(9)

import matplotlib.pyplot as plt

sample_one, sample_two = next(iter(train_ds))

plt.figure()
for n in range(9):
    ax = plt.subplot(3, 3, n+1)
    plt.plot(XGRID, sample_one[n],'-', label="one")
    plt.plot(XGRID, sample_two[n],'-',label="two")
    plt.axis("off")
plt.show()

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

TEMPERATURE = 0.25
INIT_LR = 1e-3
EPOCHS = 8
HID_SIZE = 16

def get_backbone(input_dim, channel=16):
    inputs = tf.keras.layers.Input((input_dim,), name="backbone_input")
    x = tf.expand_dims(inputs, axis=-1)
    x = tf.keras.layers.Conv1D(channel/8, kernel_size=5, strides=2, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(channel/4, kernel_size=5, strides=2, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(channel/2, kernel_size=5, strides=2, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(channel, kernel_size=5, strides=2, activation="relu")(x)
    #x = tf.keras.layers.Conv1D(32, kernel_size=2, strides=2, activation="relu", padding="same")(x)
    #x = FocalModulation(dim=16, focal_window=2, focal_level=2, focal_factor=2, proj_drop_rate=0.1)(x)
    #x = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=16, dropout=0.)(x, x)
    #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + MultiHeadAttention(d_model=18, num_heads=3, output_dim=32)(x, x, x))
    #x = EncBlock(embed_dim=32, d_model=18, num_heads=3, ff_dim=32, rate=0.)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(channel, activation="relu", name="backbone_output")(x)
    return tf.keras.Model(inputs, outputs, name="backbone")

backbone = get_backbone( INTERP_TO, channel=128)
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

#model.save_weights("./CNNModel/ckpt")

test = np.load('/content/drive/MyDrive/Colab Notebooks/SpaceScience/Dataset/test_icon_2022_10_15_half_day.npy', allow_pickle=True).item()
RI_test = test['ri'].reshape(-1,32)
RV_test = test['rv'].reshape(-1,32)
vx_test = test['ground_pars'][:,1]
AP_test = test['ground_pars'][:,2]
Ti_test = test['ground_pars'][:,3]
Hf_test= test['ground_pars'][:,4]
Of_test = test['ground_pars'][:,5]
vcor_test = test['ground_pars'][:,6]
vst_test = test['ground_pars'][:,7]

Reference = np.load('/content/drive/MyDrive/Colab Notebooks/SpaceScience/Dataset/train_IVs&pars_uniform.npy', allow_pickle=True).item()
RI_ref = Reference['ri'].reshape(-1,32)
RI_ref = RI_ref/np.mean(RI_ref[:,:4],axis=-1,keepdims=True)
RV_ref = Reference['rv'].reshape(-1,32)
vst_ref = Reference['ground_pars']['vst']
vx_ref = Reference['ground_pars']['vx']
vcor_ref = Reference['ground_pars']['v_cor_x']
Ti_ref = Reference['ground_pars']['Ti']
Of_ref = Reference['ground_pars']['O_frac']
Hf_ref = Reference['ground_pars']['H_frac']
AP_ref = Reference['ground_pars']['AP_POT']

idx, _ = np.where( np.isnan(RI_ref))
RI_ref = np.delete(RI_ref, idx, axis=0)
RV_ref = np.delete(RV_ref, idx, axis=0)
vst_ref = np.delete(vst_ref, idx, axis=0)
vx_ref = np.delete(vx_ref, idx, axis=0)
vcor_ref = np.delete(vcor_ref, idx, axis=0)
Ti_ref = np.delete(Ti_ref, idx, axis=0)
Of_ref = np.delete(Of_ref, idx, axis=0)
Hf_ref = np.delete(Hf_ref, idx, axis=0)
AP_ref = np.delete(AP_ref, idx, axis=0)

ref_ds = tf.data.Dataset.from_tensor_slices( (RV_ref, RI_ref) ).map(lambda rv, ri: tf_interp(XGRID, rv, ri), num_parallel_calls=tf.data.AUTOTUNE).batch(9)
test_ds = tf.data.Dataset.from_tensor_slices( (RV_test, RI_test) ).map(lambda rv, ri: tf_interp(XGRID, rv, ri), num_parallel_calls=tf.data.AUTOTUNE).batch(9)

ref_knn = np.concatenate([x for x in tfds.as_numpy(ref_ds)])
test_knn = np.concatenate([x for x in tfds.as_numpy(test_ds)])
index_knn = faiss.IndexFlatL2(ref_knn.shape[1])
index_knn.add(ref_knn)
distances_knn, indices_knn = index_knn.search(test_knn, k=4)

reference = model.predict(ref_ds, verbose=0)
test = model.predict(test_ds, verbose=0)
index = faiss.IndexFlatL2(reference.shape[1])
index.add(reference)
distances, indices = index.search(test, k=4)

# lookup reference label via predicted indices
votes =AP_ref[indices]
votes_knn =AP_ref[indices_knn]

plt.plot(AP_test[:4000],'k--')
plt.plot( np.mean(votes[:4000, :], axis=1), 'r')
plt.plot( np.mean(votes_knn[:4000, :], axis=1), 'b')
plt.ylim(-2.0, -1.0)

idx = 2900
plt.plot(RV_test[idx], RI_test[idx], '.')
plt.plot(RV_ref[indices[idx,0]], RI_ref[indices[idx,0]])
plt.plot(RV_ref[indices[idx,1]], RI_ref[indices[idx,1]])
plt.plot(RV_ref[indices[idx,2]], RI_ref[indices[idx,2]])
plt.plot(RV_ref[indices[idx,3]], RI_ref[indices[idx,3]])

print('AP:', AP_ref[indices[idx]], AP_test[idx])
print('vx:', vx_ref[indices[idx]], vx_test[idx])
print('Ti:', Ti_ref[indices[idx]], Ti_test[idx])
print('Of:', Of_ref[indices[idx]], Of_test[idx])