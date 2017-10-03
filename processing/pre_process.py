import bson
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
from sklearn import preprocessing
import io

n_rows = 10
n_entries_overall = 7069896
pix_x, pix_y, rgb = 180, 180, 3

Y_flattened_shape = n_rows
y = np.zeros((Y_flattened_shape, 1)).astype(int)
x_images = np.zeros((Y_flattened_shape, pix_x, pix_y, rgb))

df_categories = pd.read_csv('../input/category_names.csv', index_col='category_id')
le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()
le.fit(df_categories.index.values)


data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))

for i, d in enumerate(data):
    if i == n_rows:
        break

    y[i] = d['category_id']
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        x_images[i] = picture
        break

print()
