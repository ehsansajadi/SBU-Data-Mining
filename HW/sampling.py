import pandas as pd

df = pd.read_csv(r'C:\Users\Ehsan\Desktop\6\Data Mining\Data-Mining\Dateset\TelegramByKamyab\Life Expectancy Data.csv')

df_random_sample = df.sample(n = 50)
print(df_random_sample)

grouped = df.groupby('Country', group_keys = False).apply(lambda x: x.sample(1))
print('--------------------------------------------------------------------')
print(grouped)