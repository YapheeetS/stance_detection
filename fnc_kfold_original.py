# The majority of the code in this file and files without headers mentioning my name are taken from :https://github.com/FakeNewsChallenge/fnc-1-baseline
#this is our baseline from the fake news challenge.

#from sklearn.svm import SVC
import numpy as np
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
    X_tfidf = generate_additional_features("tfidf", h, b, "features/tfidf."+name+".npy")
    #X_embeddings = generate_additional_features("embeddings", h, b, "features/embeddings."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_overlap,X_refuting,X_belief,X_denial,X_doubt,X_fake,X_knowledge,X_negation,X_question,X_report,X_tfidf]
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

    #split them based on class
    labels =["unrelated","disagree","agree","discuss"]
    splitted = {label: [] for label in labels}
    limit = 1535  # this is the number of examples the least class "disagree"

    for stance in d.stances:
        if stance['Stance']=='unrelated':
            if len(splitted['unrelated']) < limit:
                splitted['unrelated'].append(stance)
        elif stance['Stance']=='disagree':
            if len(splitted['disagree']) < limit:
                splitted['disagree'].append(stance)
        elif stance['Stance']=='agree':
            if len(splitted['agree']) < limit:
                splitted['agree'].append(stance)
        elif stance['Stance']=='discuss':
            if len(splitted['discuss']) < limit:
                splitted['discuss'].append(stance)

    #combine and stratify
    d.stances =[]
    for i in range(0,1535) :
        d.stances.append(splitted['unrelated'][i])
        d.stances.append(splitted['disagree'][i])
        d.stances.append(splitted['agree'][i])
        d.stances.append(splitted['discuss'][i])

    return d


if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()

    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    #Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")

    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)   # to be replaced with an SVM or a NN
        #clf = SVC(max_iter=1000,tol=1e-06)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
