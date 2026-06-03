import pandas as pd
df = pd.read_csv('dataset/obesity.csv')
print(df.groupby('NObeyesdad')[['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']].mean())
