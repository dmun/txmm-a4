from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import nltk
import functions as F
import string
from pandarallel import pandarallel
from joblib import Parallel, delayed
from itertools import chain

# %%
np.random.seed(42)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
pandarallel.initialize(progress_bar=False)

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
X_train = F.extract_features(train_data).fillna(0)
X_test = F.extract_features(test_data).fillna(0)
# X_dev = F.extract_features(dev_data).fillna(0)

# %%
def perform_ablation(features):
    columns = list(chain.from_iterable(features))
    X_train_filtered = X_train.filter(items=columns)
    X_test_filtered = X_test.filter(items=columns)

    def run_clf(dropped: str | list[str]):
        X_train_dropped = X_train_filtered.drop(dropped, axis=1)
        X_test_dropped = X_test_filtered.drop(dropped, axis=1)
        # clf = RandomForestClassifier(max_depth=30, n_estimators=100, random_state=42)
        clf = SVC(kernel='linear', C=1, random_state=42)
        clf.fit(X_train_dropped, y_train)
        y_pred = clf.predict(X_test_dropped)

        score = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        return score, acc

    return np.array(Parallel(-1)(delayed(run_clf)(x) for x in features)).T


def plot_ablation_results(results, groups):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = list(string.ascii_uppercase)[: len(groups)]
    labels = [f"{x}: {group}" for x, group in zip(xs, groups)]
    ax.bar(xs, results[0], label=labels)
    ax.set_ylim(bottom=max(np.min(results) - 0.1, 0))
    ax.set_xticks(xs)
    ax.set_xlabel("Remove feature group")
    ax.set_ylabel("F-measure")
    ax.legend()


# %%
features = {
    "None":                                     [],
    "Average token length":                     ["avg_num_chars_per_token"],
    "Total number of chars":                    ["num_chars"],
    "Total number of sentences":                ["num_sents"],
    "Total number of alphabets":                ["num_alpha"],
    "Total number of punctuation marks":        ["num_punct"],
    "Total number of double punctuation marks": ["num_punct2"],
    "Total number of triple punctuation marks": ["num_punct3"],
    "Total number of contractions":             ["num_contractions"],
    "Total number of parentheses":              ["num_parentheses"],
    "Total number of emoticons":                ["num_emoticons"],
    "Total number of no-vowel words":           ["num_no_vowel_words"],
    "Average word length":                      ["avg_word_length"],
    "Total number of single quotes":            ["num_single_quotes"],
    "Total number of double quotes":            ["num_double_quotes"],
    "Total number of whitespaces":              ["num_whitespaces"],
    "Count of each function word":              [f"fn:{word}" for word in F.ENGLISH_STOP_WORDS],
    "Count of each letter":                     [f"let:{letter}" for letter in string.ascii_letters],
    "Count of each POS tag":                    [f"pos:{tag}" for tag in F.POS_TAGS],
    "Count of each phrase":                     [f"{phrase}" for phrase in F.PHRASE.keys()],
}
results = perform_ablation(features.values())
plot_ablation_results(results, features.keys())
