import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def clean_data(df):

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df.drop_duplicates(inplace=True)

    return df

df = pd.read_csv("titanic.csv")
df = clean_data(df)

# Missing values
df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

df['Sex'] = encoder.fit_transform(df['Sex'])
df['Embarked'] = encoder.fit_transform(df['Embarked'])

# Standardize numerical columns
scaler = StandardScaler()

df[['Age','Fare']] = scaler.fit_transform(
    df[['Age','Fare']]
)

print(df.head())

df.to_csv("cleaned_titanic.csv", index=False)

print("Data Cleaning Completed")