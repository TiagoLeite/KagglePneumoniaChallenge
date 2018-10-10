import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

df1 = pd.read_csv('scored_196.csv')
df2 = pd.read_csv('has_pneu.csv')

# print(df2[df2['prob'] < 0.97].count())

# print(df2)

print(df2[df2['has_pneu'] == 1].count())

df2['randNumCol'] = np.random.randint(0, 5, 1000)  # 10 percent

df_joined = pd.merge(df1, df2, on=['patientId'])

# print(df_joined[23:33])

df_joined['PredictionString'] = np.where((df2['randNumCol'] > 0) | (df_joined['has_pneu'] == 1),
                                         df_joined['PredictionString'], '')

df_joined = df_joined.drop(['has_pneu'], axis=1)
df_joined = df_joined.drop(['randNumCol'], axis=1)
df_joined = df_joined.drop(['prob'], axis=1)


df_joined.to_csv('scored_joined.csv', index=False)

