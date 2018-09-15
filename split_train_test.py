import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/train_labels.csv')

train, test = train_test_split(data, test_size=0.2)

print(len(train), len(test))

train.to_csv('data/train_val_80.csv')
test.to_csv('data/test_val_20.csv')
