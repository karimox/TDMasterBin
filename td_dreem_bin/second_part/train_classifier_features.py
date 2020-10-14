"""
Train a classifier on precomputed features
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from td_dreem_bin.utils.scores import get_all_score


def load_datasets():
    # load datasets
    import os
    from td_dreem_bin import path_repo
    save_folder = os.path.join(path_repo, "data", "processed/")

    load_path = os.path.join(save_folder, "record_datatrain_features.npz")
    npzfile = np.load(load_path)
    x_train, y_train = npzfile['x_train'], npzfile['y_train']
    load_path = os.path.join(save_folder, "record_datatest_features.npz")
    npzfile = np.load(load_path)
    x_test, y_test = npzfile['x_test'], npzfile['y_test']

    return x_train, y_train, x_test, y_test


def estimator_rfboost():
    rf_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
    gradient_pipeline = make_pipeline(HistGradientBoostingClassifier(max_iter=1000, l2_regularization=0.6,
                                                                     learning_rate=0.01, random_state=30))

    estimators = [('Random Forest', rf_pipeline),
                  ('Gradient Boosting', gradient_pipeline)]

    stacking_classifer = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return stacking_classifer


def pca_on_datasets(x_train, x_test, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train_new = pca.fit_transform(x_train)
    x_test_new = pca.fit_transform(x_test)

    return x_train_new, x_test_new


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_datasets()

    # shuffle input data
    p = np.random.permutation(len(y_train))
    x_train, y_train = x_train[p], y_train[p]
    # Dimension reduction
    x_train_new, x_test_new = x_train, x_test
    # x_train_new, x_test_new = pca_on_datasets(x_train, x_test, n_components=100)

    # select estimator
    clf = RandomForestClassifier(max_depth=10, random_state=42)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                             n_estimators=200,
                             learning_rate=0.5)
    clf = estimator_rfboost()

    # fit & predict
    print('training...')
    clf.fit(x_train_new, y_train)
    predictions = clf.predict(x_test_new)

    # scores
    all_scores = get_all_score(y_test, predictions)
    print('F1-score = %2d %%' % (all_scores['f1_score']*100))
    print('Cohen Kappa = %2d %%' % (all_scores['cohen_kappa']*100))
    print('Accuracy = %2d %%' % (all_scores['accuracy']*100))
