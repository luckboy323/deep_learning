# -*- coding: UTF-8 -*-
#numpy科学计算工具箱
import numpy as np
#使用make_classification构造1000个样本，每个样本有20个feature
from sklearn.datasets import make_classification
X, y = make_classification(1000, n_features=20, n_informative=2,
                           n_redundant=2, n_classes=2, random_state=0)
#存为dataframe格式
from pandas import DataFrame
df = DataFrame(np.hstack((X, y[:, None])),columns = range(20) + ["class"])

print df[:6]

import matplotlib.pyplot as plt
import seaborn as sns
#使用pairplot去看不同特征维度pair下数据的空间分布状况
_ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5)
plt.show()


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
_ = sns.corrplot(df, annot=False)
plt.show()