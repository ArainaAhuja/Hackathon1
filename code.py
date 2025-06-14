from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.stats import zscore
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


df = pd.read_csv('hacktrain.csv')
df.drop(columns=['ID'], inplace=True)

# All missing values are replaced with the mean for numerical columns and mode for categorical columns
# for column in df.columns:
#     if df[column].dtype == 'float64' or df[column].dtype == 'int64':
#         df[column].fillna(df[column].mean(), inplace=True)
#     else:
#         df[column].fillna(df[column].mode()[0], inplace=True)

# All outliers are removed using the IQR method


def remove_outliers(dataframe):
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        for idx, value in dataframe[column].items():
            if value > upper or value < lower:
                dataframe.at[idx, column] = np.nan
            else:
                dataframe.at[idx, column] = value


remove_outliers(dataframe=df)

for column in df.columns:  # filling missing values
    if df[column].dtype == 'float64' or df[column].dtype == 'int64':
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)

LE = LabelEncoder()
df['class'] = LE.fit_transform(df['class'])
sorted(df['class'])

X = df.drop(columns=['class'])
Y = df['class']

scaler = StandardScaler()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)
model = LogisticRegression(
    multi_class='multinomial',
    solver='newton-cg',
    max_iter=1000000,
    C=0.1,
    penalty='l2',
    random_state=42)
model.fit(X_train, Y_train)

X_train = scaler.fit_transform(X)
Y_pred = model.predict(X_test)


print(classification_report(
    Y_test,
    Y_pred,
    labels=list(range(len(LE.classes_))),
    target_names=LE.classes_
))
f1 = f1_score(Y_test, Y_pred, average='macro')
print("Macro F1 score:", f1)


test = pd.read_csv('hacktest.csv')

test.drop(columns=['ID'], inplace=True)
for column in test.columns:
    if test[column].dtype == 'float64' or test[column].dtype == 'int64':
        test[column].fillna(test[column].mean(), inplace=True)
    else:
        test[column].fillna(test[column].mode()[0], inplace=True)

testx = test[X.columns]
testx = scaler.transform(testx)
predictions = model.predict(testx)
predictions = LE.inverse_transform(predictions)
print(predictions)

result = pd.DataFrame({
    'ID': pd.read_csv('hacktest.csv')['ID'],
    'class': predictions
})

print(result.head())

result.to_csv("submission-final.csv", index=False)
