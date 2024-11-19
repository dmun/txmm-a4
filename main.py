from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import nltk
import functions as F

# %%
pd.set_option("display.max_colwidth", None)
nltk.download("punkt_tab")

# %%
train_data = pd.read_csv("./data/pan2425_train_data.csv")
dev_data = pd.read_csv("./data/pan2425_dev_data.csv")
test_data = pd.read_csv("./data/pan2425_test_data.csv")

y_train = train_data["author"]
y_dev = dev_data["author"]
y_test = test_data["author"]

print(train_data.head())


# %%
reload(F)
X_train = F.extract_features(train_data).to_numpy()
X_test = F.extract_features(test_data).to_numpy()

# %%
clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
f = f1_score(y_test, y_pred, average="weighted")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F-score: {f:.2f}")

# %%
plt.figure()

accuracies = []
f_scores = []
params = np.arange(10, 50, 5)
for p in params:
    clf = RandomForestClassifier(max_depth=p, n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f_scores.append(f1_score(y_test, y_pred, average="weighted"))
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(params, f_scores, label="f-score")
plt.plot(params, accuracies, label="accuracy")
plt.xlabel("max_depth")

plt.grid()
plt.legend()
plt.show()

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"F-score: {f:.2f}")
