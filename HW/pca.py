import pandas as pd
df = pd.read_csv(r'C:\Users\Ehsan\Desktop\6\Data Mining\Data-Mining\Dateset\TelegramByKamyab\kj (2).csv')

df = df.drop([0, 1])
df = df.drop(['Unnamed: 0', 'Unnamed: 82'], axis=1)

categorical_cols = set(df.columns) - set(df._get_numeric_data().columns)
print(categorical_cols)

df = df.drop(['شهر'], axis=1)
categorical_cols.remove('شهر')
print(categorical_cols)

categorical_subset = df[categorical_cols]
categorical_subset = pd.get_dummies(categorical_subset)
df.drop(columns=categorical_cols, inplace=True)
df = pd.concat([df, categorical_subset], axis=1)
print(df.shape)

from sklearn.decomposition import PCA
pca = PCA(0.95)
x_pca = pca.fit_transform(df)
print(x_pca.shape)

print(x_pca)