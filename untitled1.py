#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:20:00 2022

@author: chihte
"""
import pandas as pd
df = pd.read_csv('creditcard.csv')

from sklearn.model_selection import train_test_split

x = df.drop(['Time', 'Class'], axis=1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

import tensorflow as tf
tf.random.set_seed(0)
BATCH_SIZE = 100

train_features = tf.convert_to_tensor(x_train)
train_labels = tf.convert_to_tensor(y_train)
train_loader = (tf.data.Dataset.from_tensor_slices((train_features, train_labels))
                .cache()    # cache will produce exactly the same elements during each iteration through the dataset.
                            # If you wish to randomize the iteration order, make sure to call shuffle after calling cache
                .shuffle(buffer_size=15000) # This dataset fills a buffer with buffer_size elements, then randomly 
                                            # samples elements from this buffer. For perfect shuffling, a buffer size 
                                            # greater than or equal to the full size of the dataset is required.
                .batch(BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))

test_features = tf.convert_to_tensor(x_test)
test_labels = tf.convert_to_tensor(y_test)
test_loader = (tf.data.Dataset.from_tensor_slices((test_features, test_labels))
                .cache()    # cache will produce exactly the same elements during each iteration through the dataset.
                            # If you wish to randomize the iteration order, make sure to call shuffle after calling cache
                .shuffle(buffer_size=15000) # This dataset fills a buffer with buffer_size elements, then randomly 
                                            # samples elements from this buffer. For perfect shuffling, a buffer size 
                                            # greater than or equal to the full size of the dataset is required.
                .batch(BATCH_SIZE)
                .prefetch(tf.data.experimental.AUTOTUNE))

#%%
import tensorflow.keras.backend as K

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

#%%
class iModel(tf.keras.Model):
    def __init__(self):
        super(iModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation='relu') # 這步後得到 (batch_size, 128) 維度 tensor
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer2 = tf.keras.layers.Dense(1, activation='sigmoid')# 這步後得到 (batch_size, 1) 維度 tensor
    def call(self, x, training=False):
        x = self.layer1(x)
        x = self.bn1(x, training=training)
        x = self.layer2(x)
        return x
model = iModel()
model.build(input_shape=(None, x_train.shape[-1]))

from tensorflow.keras import metrics
lr = 0.001
num_epochs = 5
train_acc = metrics.Accuracy()
test_acc = metrics.Accuracy()
#BCELoss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

import tensorflow_addons as tfa
#BCELoss = tfa.losses.SigmoidFocalCrossEntropy(from_logits= False, alpha = 0.75, gamma=2.)
#BCELoss = binary_focal_loss(gamma=2., alpha=.75)

@tf.function
def train_model(x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        #loss = BCELoss(y_true, y_pred)
        loss = tf.reduce_sum(tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred))
    grads = tape.gradient(loss, model.trainable_variables)    # 使用 model.variables 這一屬性直接獲得模型中的所有變數
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    train_acc.update_state(tf.argmax(y_pred, axis=1), y_true)
    return loss

acc = []
val = []
loss = []
for epoch in range(num_epochs):
    for n, (x,y) in enumerate(train_loader):
        l = train_model(x,y)
    loss[len(loss):] = [l]
    acc[len(acc):] = [train_acc.result().numpy()]
    train_acc.result().numpy()
    
    for n, (x,y) in enumerate(test_loader):
        y_pred = model(x)
        test_acc.update_state(tf.argmax(y_pred, axis=1), y)
    val[len(val):] = [test_acc.result().numpy()]
    test_acc.result().numpy()
        
    # Show loss
    if epoch % 1 == 0:
        print(f"Epoch: {epoch} |Loss: {loss[-1]} |acc: {acc[-1]} val:{val[-1]}")

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

y_predicted = model.predict(x_test) > 0.5
mat = metrics.confusion_matrix(y_test, y_predicted)
labels = ['Legitimate', 'Fraudulent']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')

false_positive_rate, recall, thresholds = metrics.roc_curve(y_test, y_predicted, pos_label=1)
rocauc = metrics.auc(false_positive_rate, recall)
plt.title(f"AUC: {rocauc}")