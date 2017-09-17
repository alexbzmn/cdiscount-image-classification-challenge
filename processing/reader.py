import io

import bson  # this is installed with the pymongo package
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread  # or, whatever image library you prefer

# multi-threaded version
#
# NCORE = 8
# prod_to_category = mp.Manager().dict()  # note the difference
#
#
# def process(q, iolock):
#     while True:
#         d = q.get()
#         if d is None:
#             break
#         product_id = d['_id']
#         category_id = d['category_id']
#         prod_to_category[product_id] = category_id
#         for e, pic in enumerate(d['imgs']):
#             picture = imread(io.BytesIO(pic['picture']))
#             # do something with the picture, etc
#
#
# q = mp.Queue(maxsize=NCORE)
# iolock = mp.Lock()
# pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))
#
# # process the file
#
# data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))
# for c, d in enumerate(data):
#     q.put(d)  # blocks until q below its max size
#
# # tell workers we're done
#
# for _ in range(NCORE):
#     q.put(None)
#     pool.close()
#     pool.join()

data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))
prod_to_category = dict()

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id']  # This won't be in Test data
    prod_to_category[product_id] = category_id
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        # do something with the picture, etc

plt.imshow(picture)
# convert back to normal dictionary
prod_to_category = dict(prod_to_category)

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

print(prod_to_category.head())
