"""This python code was created to test the possibilities of balancing out a dataset by
flipping images. The dataset MNIST was used, but showed negative results giving off a lower
 accuracy when using flipped images. 54% accuracy for un-augmented dataset. 50% accuracy for
 augmented dataset."""
from __future__ import division
import numpy as np
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



class flipData:
    def __init__(self):
        pass

    def load(self,data_location):
        # load data
        self.data = np.load(data_location)
        return self.data['X'],self.data['y']

    def setup(self,X,y):
        self.X=X
        self.y=y
        self.N, self.D = X.shape

        labels = []
        for i in y:
            if i not in labels:
                labels.append(i)
        self.labels = labels


    def countLabel(self,label,y):
        return np.sum(y == label)

    def countAllLabels(self,y):
        count = {}
        for label in self.labels:
            count[label]=np.sum(y == label)
            print(f'Number of label {label} counted:{count[label]}')
        return count

def show_image(x, title="", clim=None, cmap=plt.cm.gray, colorbar=False):
    ax = plt.gca()
    im = ax.imshow(x.reshape((28, 28)), cmap=cmap, clim=clim)

    if len(title) > 0:
        plt.title(title)

    plt.axis('off')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.show()

data=flipData()
#MANGLER DATASÆTTET"
#X,y=data.load("mnt_bin.npz")
data.setup(X,y)
label_count=data.countAllLabels(y)
lowest_label=min(label_count, key=label_count.get)
highest_label=max(label_count, key=label_count.get)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d images : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
print(f"Accuracy of {(X_test.shape[0]-(y_test != y_pred).sum())/X_test.shape[0]}")

X_aug=X
y_aug=y

final_N_augmented=label_count[lowest_label]*2

for label in label_count.keys():
    #augNumber=label_count[highest_label]-label_count[label]
    augNumber=final_N_augmented-label_count[label]
    label_images=X[y == label]
    for i in range(min(100000,augNumber)):
        #show_image(label_images[i])
        x_new=np.fliplr(label_images[i].reshape((28, 28))).reshape(784)
        #show_image(x_new)
        X_aug=np.vstack((X_aug, x_new))
        y_aug=np.append(y_aug,label)

X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled data augmented points out of a total %d images : %d"% (X_test.shape[0], (y_test != y_pred).sum()))
print(f"Accuracy of {(X_test.shape[0]-(y_test != y_pred).sum())/X_test.shape[0]}")








