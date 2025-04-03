# -*- coding: utf-8 -*-

!pip install catboost
!pip install xgboost
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax
import lightgbm as lgb
from lightgbm import LGBMClassifier

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/rspecs.csv', sep=',')
print(type(df))
df.head()

def Labelling(Rows):
    count = 0
    for digit in Rows:
        if digit == 1:
            count += 1

    if (count > 11):
      Label="1"
    elif (6 < count <= 11):
      Label="2"
    else:
      Label="3"
    return Label

df["label"]=df.apply(Labelling, axis=1)

df.head()

label=list(df.label)
one=0
two=0
three=0
a={'1':one,'2':two,'3':three}
for i in label:
  count=0
  if i =='1':
    count+=1
    one=one+count
  elif i =='2':
    count+=1
    two=count+two
  elif i=='3':
    count+=1
    three=three+count
a={'1':one,'2':two,'3':three}
print(a)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

X = df.drop('label', axis=1)
y = df['label']

class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
y = y.map(class_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

accuracies = []
gb_accuracies=[]
cat_accuracies=[]
ada_accuracies=[]
lgb_accuracies=[]

catboost_scores = []
gradientboost_scores = []
ada_scores=[]
lgb_scores=[]
xg_scores=[]

for n in n_estimators:
    # XGBoost Classifier
    xgb_clf = XGBClassifier(n_estimators=n, random_state=42)
    xgb_clf.fit(X_train, y_train)
    xgb_predictions = xgb_clf.predict(X_test)
    accuracy = accuracy_score(y_test, xgb_predictions)
    accuracies.append(accuracy)
    xg_scores.append(f1_score(y_test, xgb_predictions, average='weighted'))

# Gradient Boosting Classifier
for n in n_estimators:
    gb_clf = GradientBoostingClassifier(n_estimators=n, random_state=42)
    gb_clf.fit(X_train, y_train)
    gb_predictions = gb_clf.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_predictions)
    gb_accuracies.append(gb_accuracy)
    gradientboost_scores.append(f1_score(y_test, gb_predictions, average='weighted'))

# CatBoost Classifier
for n in n_estimators:
    cat_clf = CatBoostClassifier(n_estimators=n, random_state=42)
    cat_clf.fit(X_train, y_train)
    cat_predictions = cat_clf.predict(X_test)
    cat_accuracy = accuracy_score(y_test, cat_predictions)
    cat_accuracies.append(cat_accuracy)
    catboost_scores.append(f1_score(y_test, cat_predictions, average='weighted'))

# AdaBoost Classifier
for n in n_estimators:
    ada_clf = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada_clf.fit(X_train, y_train)
    ada_predictions = ada_clf.predict(X_test)
    ada_accuracy = accuracy_score(y_test, ada_predictions)
    ada_accuracies.append(ada_accuracy)
    ada_scores.append(f1_score(y_test, ada_predictions, average='weighted'))

# LightGBM Classifier
for n in n_estimators:
    lgb_clf = LGBMClassifier(n_estimators=n, random_state=42)
    lgb_clf.fit(X_train, y_train)
    lgb_predictions = lgb_clf.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_predictions)
    lgb_accuracies.append(lgb_accuracy)
    lgb_scores.append(f1_score(y_test, lgb_predictions, average='weighted'))

plt.plot(n_estimators, accuracies,label='XGBoost', marker='o')
plt.plot(n_estimators, gb_accuracies,label='Gradient Boosting', marker='x')
plt.plot(n_estimators, cat_accuracies,label='CatBoost', marker='+')
plt.plot(n_estimators, ada_accuracies,label='AdaBoost', marker='*')
plt.plot(n_estimators, lgb_accuracies,label='LightGBM', marker='.')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend(loc='center right')
plt.savefig("accuracy.png")
plt.show()

plt.plot(n_estimators, catboost_scores, label='CatBoost', marker='+')
plt.plot(n_estimators, gradientboost_scores, label='Gradient Boosting',marker='x')
plt.plot(n_estimators, xg_scores, label='XGBoost',marker='o')
plt.plot(n_estimators, ada_scores, label='AdaBoost',marker='*')
plt.plot(n_estimators, lgb_scores, label='LightGBM',marker='.')
plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score')
plt.legend(loc='center right')
plt.savefig("F1 score.png")
plt.show()

tree_values = [100, 200]
leaf_values = [5,10,15]

accuracies_xgb = []
accuracies_catboost = []
accuracies_gradientboost = []
accuracies_adaboost = []
accuracies_lightgbm = []

for trees in tree_values:
    for leaves in leaf_values:
        # XGBoost Classifier
        xgb_clf = XGBClassifier(n_estimators=trees, max_depth=leaves, random_state=42)
        xgb_clf.fit(X_train, y_train)
        xgb_predictions = xgb_clf.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_predictions)
        accuracies_xgb.append(xgb_accuracy)

        # CatBoost Classifier
        catboost_clf = CatBoostClassifier(n_estimators=trees, max_depth=leaves, random_state=42)
        catboost_clf.fit(X_train, y_train)
        catboost_predictions = catboost_clf.predict(X_test)
        catboost_accuracy = accuracy_score(y_test, catboost_predictions)
        accuracies_catboost.append(catboost_accuracy)

        # Gradient Boosting Classifier
        gradientboost_clf = GradientBoostingClassifier(n_estimators=trees, max_depth=leaves, random_state=42)
        gradientboost_clf.fit(X_train, y_train)
        gradientboost_predictions = gradientboost_clf.predict(X_test)
        gradientboost_accuracy = accuracy_score(y_test, gradientboost_predictions)
        accuracies_gradientboost.append(gradientboost_accuracy)

        # Ada Boosting Classifier
        adaboost_clf = AdaBoostClassifier(n_estimators=trees, random_state=42)
        adaboost_clf.fit(X_train, y_train)
        adaboost_predictions = adaboost_clf.predict(X_test)
        adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
        accuracies_adaboost.append(adaboost_accuracy)

        # LightGBM Classifier
        lightgbm_clf = LGBMClassifier(n_estimators=trees, random_state=42)
        lightgbm_clf.fit(X_train, y_train)
        lightgbm_predictions = lightgbm_clf.predict(X_test)
        lightgbm_accuracy = accuracy_score(y_test, lightgbm_predictions)
        accuracies_lightgbm.append(lightgbm_accuracy)


labels = [f"T:{trees}, L:{leaves}" for trees in tree_values for leaves in leaf_values]
x_pos = np.arange(len(labels))
#print(labels)
bar_width = 0.15

offset = bar_width * 2

plt.bar(x_pos - offset, accuracies_adaboost, width=bar_width, align='center', alpha=0.8, label='AdaBoost', hatch='.')


plt.bar(x_pos - bar_width, accuracies_xgb, width=bar_width, align='center', alpha=0.8, label='XGBoost',hatch='x')


plt.bar(x_pos, accuracies_catboost, width=bar_width, align='center', alpha=0.8, label='CatBoost', hatch='-')


plt.bar(x_pos + bar_width, accuracies_gradientboost, width=bar_width, align='center', alpha=0.8, label='Gradient Boosting',hatch='o')


plt.bar(x_pos + 2 * bar_width, accuracies_lightgbm, width=bar_width, align='center', alpha=0.8, label='LightGBM',hatch='*')


plt.xlabel('Tree (T) and Leaf (L) Values')
plt.ylabel('Accuracy')
#plt.title('Comparison of Boosting Algorithms')


plt.xticks(x_pos, labels, rotation='vertical')


plt.legend(loc='center right')


plt.tight_layout()
plt.savefig("Boosting algorithms with different trees and leaves.png")
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

X = df.drop('label', axis=1)
y = df['label']

class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
y = y.map(class_mapping)

num_classes = len(set(y))
y_true = label_binarize(y, classes=range(num_classes))

models = {
    'Gradient Boosting':GradientBoostingClassifier(),
    'CatBoost': CatBoostClassifier(),
    'LightGBM': LGBMClassifier(),
    'XGBoost': XGBClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

precision = dict()
recall = dict()
thresholds = dict()

for model_name, model in models.items():
    model.fit(X, y)

    predicted_probabilities = model.predict_proba(X)

    precision[model_name], recall[model_name], thresholds[model_name] = precision_recall_curve(
        y_true.ravel(), predicted_probabilities.ravel()
    )


plt.figure()

for model_name in models.keys():
    plt.step(recall[model_name], precision[model_name], where='post', label=model_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig("PR curve.png")
plt.show()

import time

catboost = CatBoostClassifier(loss_function='MultiClass', logging_level='Silent')
lightgbm = LGBMClassifier(objective='multiclass', random_state=42)
xgboost = XGBClassifier(objective='multi:softmax', random_state=42)
adaboost = AdaBoostClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)


start_time = time.time()
catboost.fit(X_train, y_train)
catboost_time = time.time() - start_time

start_time = time.time()
lightgbm.fit(X_train, y_train)
lightgbm_time = time.time() - start_time

start_time = time.time()
xgboost.fit(X_train, y_train)
xgboost_time = time.time() - start_time

start_time = time.time()
adaboost.fit(X_train, y_train)
adaboost_time = time.time() - start_time

start_time = time.time()
gradient_boosting.fit(X_train, y_train)
gradient_boosting_time = time.time() - start_time

classifiers = ['CatBoost', 'LightGBM', 'XGBoost', 'AdaBoost', 'Gradient Boosting']
training_times = [catboost_time, lightgbm_time, xgboost_time, adaboost_time, gradient_boosting_time]
hatches = ['*', '\\', 'x', '.', '+']

plt.bar(classifiers, training_times,hatch=hatches)
plt.xlabel('Classifiers')
plt.ylabel('Training Time (seconds)')
plt.savefig("Traning times.png")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


X = df.drop('label', axis=1)
y = df['label']


label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)


classifiers = [
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    xgb.XGBClassifier(),
    CatBoostClassifier(),
    lgb.LGBMClassifier()
]

plt.figure(figsize=(8, 6))

markers = ['o', '.', 'x', '+','1']

for classifier in classifiers:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)


    plt.plot(fpr, tpr, label=f'{classifier.__class__.__name__} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("roc.png")
plt.show()
