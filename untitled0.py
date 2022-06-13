"""
#from IPython import get_ipython
#get_ipython().magic('reset -f')
#get_ipython().run_line_magic('reset', '-f')
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
        loss = CCEloss(y_true, y_pred)
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

#%%
# The function to generate interpolated input features along a linear path at alpha intervals 
# between the baseline input and a example imput we want to explain the model
def interpolate_inputs(baseline,
                       inp,
                       alphas):
    alphas_x = alphas[:, tf.newaxis] # [m_steps+1, 1]
    #print(alphas_x.shape)
    baseline_x = tf.expand_dims(baseline, axis=0) # [1, num_feat]
    input_x = tf.expand_dims(inp, axis=0) # [1, num_feat]
    delta = input_x - baseline_x
    #print(delta.shape)
    inputs = baseline_x +  alphas_x * delta
    #print(xs.shape)
    return inputs


# TensorFlow makes computing gradients easy for you with a tf.GradientTape
def compute_gradients(inputs, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        logits = model(inputs)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, inputs)


@tf.function
def one_batch(baseline, inp, alpha_batch, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_inputs(baseline = baseline,
                                                       inp = inp,
                                                       alphas = alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(inputs = interpolated_path_input_batch,
                                       target_class_idx = target_class_idx)
    return gradient_batch


# The function takes the gradients of the predicted probability of the target class 
# with respect to the interpolated images between the baseline and the original image.
def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(baseline,
                         inputs,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
    # Generate m_steps intervals for integral_approximation() below.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    
    # Collect gradients.    
    gradient_batches = []
    
    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        gradient_batch = one_batch(baseline, inputs, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)
        
    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)
    
    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)
    
    # Scale integrated gradients with respect to input.
    integrated_gradients = (inputs - baseline) * avg_gradients
    
    return integrated_gradients


# Establish a baseline for Integrated Gradient
# A baseline is an input vector used as a starting point for calculating feature importance
# Here, we use median values for each feature
baseline = tf.convert_to_tensor([ np.median(x_train[col]) for col in df_x.columns ], dtype=np.float32)

TAR = 1
for idx in np.where(y_train == TAR)[-1]:
    ig_attributions = integrated_gradients(baseline = baseline,
                                           inputs = tf.cast(train_features[idx], tf.float32),
                                           target_class_idx = TAR,
                                           m_steps = 50)
    plt.plot( ig_attributions, '.')