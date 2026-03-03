import pandas as pd

df = pd.read_csv('data/Shakespeare_data.csv')

# get rid of any non-line data
df = df.dropna(subset=['Player'])

# remove players with 2 lines or less
df = df[df['Player'].map(df['Player'].value_counts()) > 2]


# value counts
print(f"there are {df['Player'].nunique()} unique players")
print(df.head())

df.to_csv('data/Shakespeare_cleaned.csv', index=False)