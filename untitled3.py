#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:27:14 2022

@author: chihte
"""

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
tf.random.set_seed(0)
train_features = tf.convert_to_tensor( x_train )
train_labels = tf.convert_to_tensor( OneHotEncoder(y_train.to_numpy(), num_classes=2), dtype=tf.float32) # [:, num_classes]

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
import tensorflow.keras.backend as K
class BinararyFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha_tensor, gamma=2.):
        super(BinararyFocalLoss, self).__init__()
        self.alpha = alpha_tensor
        self.gamma = gamma
    
    def BCE_elementwise(self, y_true, y_pred):
        #log_pred = tf.clip_by_value(tf.math.log(y_pred), -100, 100)
        #log_pred_rev = tf.clip_by_value(tf.math.log(1-y_pred), -100, 100)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        log_pred = tf.math.log(y_pred + 1e-7)
        log_pred_rev = tf.math.log(1-y_pred + 1e-7)
        return -( y_true*log_pred + (1-y_true)*log_pred_rev)
    
    def call(self, y_true, y_pred):
        sample_weight = tf.reduce_sum( self.alpha*y_true, axis=-1)[:,tf.newaxis]
        #print( sample_weight )
        #BCE_loss = self.BCE_elementwise(y_true, y_pred)
        BCE_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        #print( BCE_loss )
        prob = tf.math.exp(-BCE_loss) # prevents nans when probability 0
        #print( prob )
        F_loss = sample_weight * (1-prob)**self.gamma * BCE_loss # alpha broadcasted
        #print( F_loss )
        F_loss = tf.math.reduce_mean(F_loss, axis=-1) # average over feature dimension
        return F_loss # !!!IMPORTANT!!! tf.keras.losses.Loss will later return the average over batch


#%%
class iModel(tf.keras.Model):
    def __init__(self):
        super(iModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation='relu') # 這步後得到 [batch_size, 128] 維度 tensor
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 這個 FFN 輸出跟我們 target class 一樣大的 logits 數，等通過 softmax 就代表每個 class 的出現機率 !!!
        self.layer2 = tf.keras.layers.Dense(2) # [batch_size, num_classes] 維度 tensor
    def call(self, x, training=False):
        x = self.layer1(x)
        x = self.bn1(x, training=training)
        # 最後在通過一個 linear layer 讓輸出變成只有目標 class 數目
        x = self.layer2(x) # |outputs| : [batch_size, num_classes] 維度 tensor
        return tf.nn.softmax(x) # IMPORTANT!!! softmax performed on the last dimension

model = iModel()
model.build(input_shape=(None, x_train.shape[-1]))

#from tensorflow.keras import metrics
#acc_meter = metrics.Accuracy()

lr = 0.001
num_epochs = 10
#CCEloss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # from_logits=False 因為 softmax 輸出是機率分佈
CCEloss = BinararyFocalLoss( tf.convert_to_tensor([0.1, 0.9]), gamma=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

# Weight Decay Regularization
#import tensorflow_addons as tfa
#MyAdamW = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
#optimizer = MyAdamW(weight_decay=0.001, learning_rate=lr)


@tf.function
def train_model(x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)  # Logits for this minibatch
        loss = CCEloss(y_true, y_pred)
    #acc_meter.update_state(tf.argmax(y_pred, axis=1), y_true)
    grads = tape.gradient(loss, model.trainable_variables)  # having BatchNormalization so not to use .variables
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
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