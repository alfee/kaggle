from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def preprocessing(data):
    # 性別を数値化
    data.loc[data["Sex"] == "male", "EncodedSex"] = 0
    data.loc[data["Sex"] == "female", "EncodedSex"] = 1
    data["FillteredAge"] = data["Age"] < 30
    data.loc[data["Embarked"] == "C", "EncodedEmbarked"] = 0
    data.loc[data["Embarked"] == "Q", "EncodedEmbarked"] = 1
    data.loc[data["Embarked"] == "S", "EncodedEmbarked"] = 2
    data["EncodedEmbarked"].fillna(-1)
    data.loc[data["EncodedEmbarked"].isnull(), "EncodedEmbarked"] = -1
    data["FillteredFare"] = data["Fare"] < 100


train = pd.read_csv("./datasets/train.csv")
test = pd.read_csv("./datasets/test.csv")

preprocessing(train)
preprocessing(test)

label_name = "Survived"

feature_names = ["EncodedSex", "FillteredAge",
                 "SibSp", "Parch", "Pclass", "FillteredFare", "EncodedEmbarked"]

x_train = train[feature_names]
y_train = train[label_name]
print(x_train)

x_test = test[feature_names]

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print(acc_random_forest)

# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(x_train, y_train)
# y_pred = decision_tree.predict(x_test)
# acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
# print(acc_decision_tree)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": y_pred
})

submission.to_csv("./result/submission.csv", index=False)
