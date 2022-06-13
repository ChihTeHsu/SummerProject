#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:45:35 2022

@author: chihte
"""
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-f')

import numpy as np
def OneHotEncoder(labels, num_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(labels).reshape(-1) # reshape is here to deal with input as colume vector
    return np.eye(num_classes)[targets]

import pandas as pd
df = pd.read_csv('creditcard.csv')

from sklearn.model_selection import train_test_split
df_x = df.drop(['Time', 'Class'], axis=1)
df_y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, stratify=df_y, random_state=0)

import tensorflow as tf
train_features = tf.convert_to_tensor( x_train )
train_labels = tf.convert_to_tensor( OneHotEncoder(y_train.to_numpy(), num_classes=2) )

BATCH_SIZE = 100
train_loader = (tf.data.Dataset.from_tensor_slices((train_features, train_labels))
                .cache()    # cache will produce exactly the same elements during each iteration through the dataset.
                            # If you wish to randomize the iteration order, make sure to call shuffle after calling cache
                .shuffle(buffer_size=15000) # This dataset fills a buffer with buffer_size elements, then randomly 
                                            # samples elements from this buffer. For perfect shuffling, a buffer size 
                                            # greater than or equal to the full size of the dataset is required.
                .batch(BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))

#%%
def roc_auc_score(y_pred, y_true):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))

#%%
class iModel(tf.keras.Model):
    def __init__(self):
        super(iModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation='relu') # 這步後得到 (batch_size, 128) 維度 tensor
        # 這個 FFN 輸出跟我們 target class 一樣大的 logits 數，等通過 softmax 就代表每個 class 的出現機率 !!!
        self.layer2 = tf.keras.layers.Dense(2) #(batch_size, num_classes) 維度 tensor
    def call(self, x):
        x = self.layer1(x)
        # 最後在通過一個 linear layer 讓輸出變成只有目標 class 數目
        x = self.layer2(x) # |outputs| : (batch_size, num_classes) 維度 tensor
        #return tf.nn.softmax(x) # default softmax performed on the last dimension
        return x

model = iModel()
model.build(input_shape=(None, x_train.shape[-1]))

#from tensorflow.keras import metrics
#acc_meter = metrics.Accuracy()

lr = 0.001
num_epochs = 10
#CCEloss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # from_logits=False 因為 softmax 輸出是機率分佈
CCEloss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def train_model(x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        #loss = CCEloss(y_true, y_pred)
        loss = roc_auc_score(y_pred, y_true)
    #acc_meter.update_state(tf.argmax(y_pred, axis=1), y_true)
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 這一屬性直接獲得模型中的所有變數
    optimizer_discriminator.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return loss

for epoch in range(num_epochs):
    for n, (x,y) in enumerate(train_loader):
        loss=train_model(x,y)
        # Show loss
        if epoch % 1 == 0 and n == BATCH_SIZE - 1:
            print(f"Epoch: {epoch} |Loss: {loss}")
            #acc_meter.reset_states()

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

probs = tf.nn.softmax(model.predict(x_test), axis=-1) # use .predict when input is dataframe
y_predicted = np.argmax( probs, axis=-1) # take Probs of each class, use argmax to define prediction
mat = metrics.confusion_matrix( y_test, y_predicted )
labels = ['Legitimate', 'Fraudulent']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')

false_positive_rate, recall, thresholds = metrics.roc_curve(y_test, y_predicted, pos_label=1)
rocauc = metrics.auc(false_positive_rate, recall)
plt.title(f"AUC: {rocauc}")