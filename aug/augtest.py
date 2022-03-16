import numpy as np
import os
from utils.input_data import read_data_sets
import utils.datasets as ds
import utils.augmentation as aug
import utils.helper as hlp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import utils
import pandas as pd
#import tensorflow_decision_forests as tfdf

dataset = "CBF"

nb_class = ds.nb_classes(dataset)
nb_dims = ds.nb_dims(dataset)

# Load Data
train_data_file = os.path.join("data", dataset, "%s_TRAIN.tsv"%dataset)
test_data_file = os.path.join("data", dataset, "%s_TEST.tsv"%dataset)

"""
data = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")

df = pd.DataFrame(data, columns =['x_train', 'y_train', 'x_test', 'y_test'])
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(df)
model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)

print(model.summary())
"""
x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")

y_train = ds.class_offset(y_train, dataset)
y_test= ds.class_offset(y_test, dataset)
nb_timesteps = int(x_train.shape[1] / nb_dims)
input_shape = (nb_timesteps , nb_dims)

x_train_max = np.max(x_train)
x_train_min = np.min(x_train)
x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
# Test is secret
x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

# How do we fit classifier with the aug shape??
"""
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
model_score = clf.score(x_test, y_test)
print("Unaugmented data score: ", model_score)

clfJitter = RandomForestClassifier(max_depth=2, random_state=0)
clfJitter.fit(aug.jitter(x_train), aug.jitter(y_train))
model_score_jitter = clfJitter.score(aug.jitter(x_test), aug.jitter(y_test))
print("Jitter score: ", model_score_jitter)


fig, axs = plt.subplots(2, 2)
steps = np.arange(x_train[0].shape[0])
axs[0, 0].plot(steps, x_train[0])
axs[0, 0].set_title('Original data')
axs[0, 1].plot(x_train[0], aug.jitter(x_train)[0])
axs[0, 1].set_title('Jittering')
axs[1, 0].plot(x_train[0], aug.scaling(x_train)[0])
axs[1, 0].set_title('Scaling')
axs[1, 1].plot(x_train[0], aug.permutation(x_train)[0])
axs[1, 1].set_title('Permutation')

#for ax in axs.flat:
#    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
"""
hlp.plot1d(x_train[0], title="Original Data")

hlp.plot1d(x_train[0], aug.jitter(x_train)[0], title="Jitter Augmentation")

hlp.plot1d(x_train[0], aug.scaling(x_train)[0], title="Scaling Augmentation")

hlp.plot1d(x_train[0], aug.permutation(x_train)[0])

hlp.plot1d(x_train[0], aug.magnitude_warp(x_train)[0])

hlp.plot1d(x_train[0], aug.time_warp(x_train)[0])

hlp.plot1d(x_train[0], aug.rotation(x_train)[0])

hlp.plot1d(x_train[0], aug.window_slice(x_train)[0])

hlp.plot1d(x_train[0], aug.window_warp(x_train)[0])

hlp.plot1d(x_train[0], aug.spawner(x_train, y_train)[0])

hlp.plot1d(x_train[0], aug.wdba(x_train, y_train)[0])

hlp.plot1d(x_train[0], aug.random_guided_warp(x_train, y_train)[0])

hlp.plot1d(x_train[0], aug.discriminative_guided_warp(x_train, y_train)[0])
