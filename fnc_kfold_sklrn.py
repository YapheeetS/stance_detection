


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, generate_baseline_feats,generate_additional_features
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = generate_baseline_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_belief = generate_additional_features("expanded_cue_words/belief.txt", h, b, "features/belief."+name+".npy")
    X_denial =generate_additional_features("expanded_cue_words/denial.txt", h, b, "features/denial."+name+".npy")
    X_doubt =generate_additional_features("expanded_cue_words/doubt.txt", h, b, "features/doubt."+name+".npy")
    X_fake =generate_additional_features("expanded_cue_words/fake.txt", h, b, "features/fake."+name+".npy")
    X_knowledge =generate_additional_features("expanded_cue_words/knowledge.txt", h, b, "features/knowledge."+name+".npy")
    X_negation =generate_additional_features("expanded_cue_words/negation.txt", h, b, "features/negation."+name+".npy")
    X_question =generate_additional_features("expanded_cue_words/question.txt", h, b, "features/question."+name+".npy")
    X_report =generate_additional_features("expanded_cue_words/report.txt", h, b, "features/report."+name+".npy")
    X_refuting = generate_baseline_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = generate_baseline_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = generate_baseline_feats(hand_features, h, b, "features/hand."+name+".npy")

    #X_embeddings = generate_additional_features("embeddings", h, b, "features/embeddings."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_overlap,X_refuting,X_belief,X_denial,X_doubt,X_fake,X_knowledge,X_negation,X_question,X_report]
    y= np.array(y)
    return X,y

def get_balanced_dataset():
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    # Load the competition dataset
    competition_dataset = DataSet("competition_test")

    # Merging the train, test to train the chosen model on the full dataset
    d.articles.update(competition_dataset.articles)
    d.stances.extend(competition_dataset.stances)

    # split them based on class
    labels = ["unrelated", "disagree", "agree", "discuss"]
    splitted = {label: [] for label in labels}
    limit = 1535  # this is the number of examples the least class "disagree"

    for stance in d.stances:
        if stance['Stance'] == 'unrelated':
            if len(splitted['unrelated']) < limit:
                splitted['unrelated'].append(stance)
        elif stance['Stance'] == 'disagree':
            if len(splitted['disagree']) < limit:
                splitted['disagree'].append(stance)
        elif stance['Stance'] == 'agree':
            if len(splitted['agree']) < limit:
                splitted['agree'].append(stance)
        elif stance['Stance'] == 'discuss':
            if len(splitted['discuss']) < limit:
                splitted['discuss'].append(stance)

    # combine and stratify
    d.stances = []
    for i in range(0, 1535):
        d.stances.append(splitted['unrelated'][i])
        d.stances.append(splitted['disagree'][i])
        d.stances.append(splitted['agree'][i])
        d.stances.append(splitted['discuss'][i])

    return d


if __name__ == "__main__":
    check_version()
    parse_params()


    d = get_balanced_dataset()
    K = 10
    X,y = generate_features(d.stances,d,"full")

    skf = StratifiedKFold(n_splits=K)
    skf.get_n_splits(X, y)
    print(skf)

    best_score = 0
    best_fold = None

    StratifiedKFold(n_splits=K, random_state=None, shuffle=False)

    # Classifier for each fold
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)   # to be replaced with an SVM or a NN
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(train_index) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf


