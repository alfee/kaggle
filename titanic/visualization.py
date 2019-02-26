import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("./datasets/train.csv", index_col = "PassengerId")

sns.set()
# sns.countplot(data=train, x="Fare", hue="Survived")
sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)


plt.show()
