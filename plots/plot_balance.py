import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
train = pd.read_csv("../datasets/train.csv", encoding="latin-1")


column = train.iloc[:,2:].sum()
#plot
plt.figure(figsize=(8,4))
ax = sns.barplot(column.index, column.values, alpha=0.8)
plt.title("Balance of classes")
plt.ylabel('Number of occurences', fontsize=13)
plt.xlabel('Quantity per class', fontsize=13)
#adding the text labels
rects = ax.patches
labels = column.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


