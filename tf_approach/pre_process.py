import io
import warnings

import bson
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
from sklearn import preprocessing

warnings.filterwarnings("ignore")

data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))
# read bson file into pandas DataFrame
with open('../input/train_example.bson', 'rb') as b:
    df = pd.DataFrame(bson.decode_all(b.read()))

for e, pic in enumerate(df['imgs'][0]):
    picture = imread(io.BytesIO(pic['picture']))
    pix_x, pix_y, rgb = picture.shape

Y_flattened_shape = len(df.index)  # cols of data in train set
X_ids = np.zeros((Y_flattened_shape, 1)).astype(int)
Y = np.zeros((Y_flattened_shape, 1)).astype(int)  # category_id for each row
X_images = np.zeros((Y_flattened_shape, pix_x, pix_y, rgb))  # m images are 180 by 180 by 3

print("Examples:", Y_flattened_shape)
print("Dimensions of Y: ", Y.shape)
print("Dimensions of X_images: ", X_images.shape)

# prod_to_category = dict()
i = 0
for c, d in enumerate(data):
    X_ids[i] = d['_id']
    Y[i] = d['category_id']
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
    X_images[i] = picture  # add only the last image
    i += 1

# Lets take a look at the category names supplied to us:
df_categories = pd.read_csv('../input/category_names.csv', index_col='category_id')

count_unique_cats = len(df_categories.index)

print("There are ", count_unique_cats, " unique categories to predict. E.g.")

# using a label encoder, and binarizer to convert all unique category_ids to have a column for each class
le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()

le.fit(df_categories.index.values)
y_encoded = le.transform(Y)

lb.fit(y_encoded)
Y_flat = lb.transform(y_encoded)

# redimension X for our model
X_flat = X_images.reshape(X_images.shape[0], -1)
Y_flat = Y_flat
X_flattened_shape = X_flat.shape[1]
Y_flattened_shape = Y_flat.shape[1]

# Scale RGB data for learning
X_flat = X_flat / 255
# print results
print("X Shape =", X_flat.shape, "Y Shape =", Y_flat.shape, "m = ", X_flattened_shape, "n classes found in test data=",
      Y_flattened_shape)

X_train = X_flat
y_train = Y_flat
