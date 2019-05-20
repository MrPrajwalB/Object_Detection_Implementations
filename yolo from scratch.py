import numpy as np
import cv2
import pandas as pd

cds=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Downloads\\keras-frcnn-master\\keras-frcnn-master\\fin.csv', header=None))
names=cds[:,0].astype(str)
pts=cds[:,1:5].astype(int)
pts[:,[2,1]] = pts[:,[1,2]]

def get_mp(pt):
    x_mid=int((pt[0]+pt[1])/2)
    y_mid=int((pt[2]+pt[3])/2)
    return (x_mid, y_mid)

def is_present(y1,y2,x1,x2,pt,siz):
    xm,ym=get_mp(pt)
    if((y1<=ym<=y2) and (x1<=xm<=x2)):
        flag=1
        by=(ym-y1)/siz
        bx=(xm-x1)/siz
        bw=(pt[3]-pt[2])/siz
        bh=(pt[1]-pt[0])/siz
    else:
        flag=0
        bbox=np.random.rand(4,)
        bx,by=bbox[0],bbox[1]
        bw,bh=bbox[2],bbox[3]
    target=np.array([flag,bx,by,bw,bh])
    target=np.resize(target,(1,5))
    return target


def get_img_param(img, siz, pt):
    w=img.shape[1]
    h=img.shape[0]
    size_x=int(w/siz)
    size_y=int(h/siz)
    target=[]
    ind_x=np.arange(0,w+siz,siz)
    ind_y=np.arange(0,h+siz,siz)
    for j in range(size_y):
        for i in range(size_x):
            targ=is_present(ind_y[j],ind_y[j+1],ind_x[i],ind_x[i+1],pt,siz)
            target.append(targ)
    return target, size_x, size_y      
images=[]
labels=[]
for k in range(len(names)):
    name=names[k].replace('/content/drive/My Drive/keras-frcnn-master/data/','')
    path='C:\\Users\\Prajwal\\Downloads\\keras-frcnn-master\\keras-frcnn-master\\data\\'+name
    print(path)
    img=cv2.imread(path)
    pt=pts[k,:]
    ftarget, wx, hy= get_img_param(img,80,pt)
    ftarget=np.concatenate(ftarget)
    ftarget=np.reshape(ftarget,(hy,wx,5))
    images.append(img)
    labels.append(ftarget)

X=np.concatenate(images)
y=np.concatenate(labels)
X=np.reshape(X,(589,512,640,3))
y=np.reshape(y,(589,6,8,5))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.15)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
regressor = Sequential()

# Step 1 - Convolution
regressor.add(Conv2D(32, (3, 3), input_shape = (512, 640, 3), activation = 'relu'))

# Step 2 - Pooling
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
#regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(Conv2D(32, (2, 2), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Dropout(0.3))
# Adding a second convolutional layer
# Step 3 - Flattening

# Step 4 - Full connection
regressor.add(Dense(units = 32, activation = 'relu'))
regressor.add(Dense(units = 8, activation = 'relu'))
regressor.add(Dense(units = 5, activation = 'linear'))

# Compiling the CNN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
print(regressor.summary())
regressor.fit(X_train, y_train,validation_data=(X_test,y_test), batch_size =8, epochs =100)
y_pred=regressor.predict(X_test)
y_pred1=y_pred[1,:,:,:]
y_test1=y_test[1,:,:,:]