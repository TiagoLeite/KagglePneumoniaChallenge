""" It generates one fake bounding box for each healthy lung image (left side only)
 Results on file "train_labels_fake.csv"
 Originally bounded boxes images stay the same"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)

data = pd.read_csv('data/stage_1_train_labels.csv')

print(data[:10])

print(np.isnan(data['x']).sum())

# print(data[data['x'] > 512].describe()
# print(data['x'].value_counts()[:][:10])

data['Target'] = np.where(data['Target'] == 0, "No", "Yes")

data['x'] = data['x'].fillna(pd.Series(np.random.normal(231, 96, 30000)))
data['y'] = data['y'].fillna(pd.Series(np.random.normal(364, 151, 30000)))
data['width'] = data['width'].fillna(pd.Series(np.random.normal(231, 96, 30000)))
data['height'] = data['height'].fillna(pd.Series(np.random.normal(364, 151, 30000)))

data['x'] = data['x'].apply(lambda x: int(x))
data['y'] = data['y'].apply(lambda x: int(x))
data['width'] = data['width'].apply(lambda x: int(x))
data['height'] = data['height'].apply(lambda x: int(x))


print(np.isnan(data['x']).sum())
print(data[:10])

data.to_csv('train_labels_fake.csv', index=False)

# print(data[:10])
