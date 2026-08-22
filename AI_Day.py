import pandas as pd  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingImports]

df = pd.read_csv("titanic.csv")
print(df.shape)          # (800, 9)
df.head(); df.info()

from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingImports]
X = df[["Age", "Fare"]]; y = df["Survived"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

scaler = StandardScaler().fit(Xtr)     # learn mean & SD from TRAIN only
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)
print("train means after scaling:", Xtr_s.mean(axis=0).round(3))   # ≈ [0, 0]
print("train SDs   after scaling:", Xtr_s.std(axis=0).round(3))    # ≈ [1, 1]

enc = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
print(enc.shape)              # how many columns now?
print([c for c in enc.columns if c not in df.columns])  

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer # Import SimpleImputer

feats = ["Pclass","Age","SibSp","Parch","Fare","Sex_male","Embarked_Q","Embarked_S"]
X, y = enc[feats], enc["Survived"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# Create a pipeline with SimpleImputer for missing values, StandardScaler, and LogisticRegression
full_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000))
full = full_pipeline.fit(Xtr, ytr)   
print("with Sex   :", round(accuracy_score(yte, full.predict(Xte)), 3))

no_sex = [f for f in feats if f != "Sex_male"]

# Create a separate pipeline for the 'no_sex' case
no_sex_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000))
ns = no_sex_pipeline.fit(Xtr[no_sex], ytr)
print("without Sex:", round(accuracy_score(yte, ns.predict(Xte[no_sex])), 3))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

num = ["Age", "Fare", "SibSp", "Parch"]
cat = ["Sex", "Embarked", "Pclass"]

num_pipe = Pipeline([("fill", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
cat_pipe = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])
prep = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])

model = Pipeline([("prep", prep),
                  ("clf", LogisticRegression(max_iter=1000))])

X = df[num + cat]; y = df["Survived"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
model.fit(Xtr, ytr)                     # scales, encodes AND trains — train only
print("test accuracy:", round(model.score(Xte, yte), 3))