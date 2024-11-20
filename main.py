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
train_dataset = pd.read_csv("./data/pan2425_train_data.csv")
dev_dataset = pd.read_csv("./data/pan2425_dev_data.csv")
test_dataset = pd.read_csv("./data/pan2425_test_data.csv")
y_train = train_dataset["author"]
y_dev = dev_dataset["author"]
y_test = test_dataset["author"]
print(train_dataset.head())

# %%
reload(F)
train_data = {
    "X": F.extract_features(train_dataset),
    "y": y_train,
}
dev_data = {
    "X": F.extract_features(dev_dataset),
    "y": y_dev,
}
test_data = {
    "X": F.extract_features(test_dataset),
    "y": y_test,
}


# %%
def perform_ablation(model, train_data, test_data, features, **kwargs):
    columns = list(chain.from_iterable(features))
    X_train = train_data["X"].filter(items=columns)
    X_test = test_data["X"].filter(items=columns)

    def run_clf(dropped: str | list[str]):
        X_train_dropped = X_train.drop(dropped, axis=1)
        X_test_dropped = X_test.drop(dropped, axis=1)
        clf = model(**kwargs)
        clf.fit(X_train_dropped, train_data["y"])
        y_pred = clf.predict(X_test_dropped)

        score = f1_score(test_data["y"], y_pred, average="weighted")
        acc = accuracy_score(test_data["y"], y_pred)
        return score, acc

    return np.array(Parallel(-1)(delayed(run_clf)(x) for x in features)).T


def plot_ablation_results(results, groups):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = list(string.ascii_uppercase)[: len(groups)]
    labels = [f"{x}: {group}" for x, group in zip(xs, groups)]
    ax.axhline(y=results[0][0], color="lightgrey", linestyle="--")
    ax.bar(xs, results[0], label=labels)
    ax.set_ylim(bottom=max(np.min(results) - 0.1, 0))
    ax.set_xticks(xs)
    ax.set_xlabel("Remove feature group")
    ax.set_ylabel("F-measure")
    ax.legend()


# %%
features = {
    "None": [],
    "Average token length": ["avg_num_chars_per_token"],
    "Total number of chars": ["num_chars"],
    "Total number of sentences": ["num_sents"],
    "Total number of alphabets": ["num_alpha"],
    "Total number of punctuation marks": ["num_punct"],
    "Total number of double punctuation marks": ["num_punct2"],
    "Total number of triple punctuation marks": ["num_punct3"],
    "Total number of contractions": ["num_contractions"],
    "Total number of parentheses": ["num_parentheses"],
    "Total number of emoticons": ["num_emoticons"],
    "Total number of no-vowel words": ["num_no_vowel_words"],
    "Average word length": ["avg_word_length"],
    "Total number of single quotes": ["num_single_quotes"],
    "Total number of double quotes": ["num_double_quotes"],
    "Total number of whitespaces": ["num_whitespaces"],
    "Count of each function word": [f"fn:{word}" for word in F.ENGLISH_STOP_WORDS],
    "Count of each letter": [f"let:{letter}" for letter in string.ascii_letters],
    "Count of each POS tag": [f"pos:{tag}" for tag in F.POS_TAGS],
    # "Count of each phrase":                     [f"{phrase}" for phrase in F.PHRASE.keys()],
}

svc_args = {"kernel": "linear", "C": 1, "random_state": 42}
svc_results = perform_ablation(
    SVC,
    train_data,
    test_data,
    features.values(),
    **svc_args,
)

rfc_args = {"max_depth": 30, "n_estimators": 100, "random_state": 42}
rfc_results = perform_ablation(
    RandomForestClassifier,
    train_data,
    test_data,
    features.values(),
    **rfc_args,
)

# %%
plot_ablation_results(svc_results, features.keys())
plot_ablation_results(rfc_results, features.keys())
