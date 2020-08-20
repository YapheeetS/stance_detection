###########################################################################################################################
#  This file is the main file used to run the trained model either on new data or to train it on the whole dataset        #
#  including (train +test). in case it is used for prediction, it first loads the model, then extracts features, then     #
#   predicts the labels. In case it is used to train the model, it loads both the train and test ds, then merges them,    #
#  then extracts features, then the model is fit to the data.                                                             #                                                                           #
#  ______________________________________________________________________________________________________________________ #
#  Author : Israa Qasim Jaradat ,  University of Texas at Arlington                                                       #
###########################################################################################################################


from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering_depoly import *
from utils.dataset import DataSet
#from utils.score import LABELS
from google_search import do_research
from utils.system import parse_params, check_version
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

import nltk
nltk.download("punkt")
nltk.download("wordnet")


class Source:
    def __init__(self):
        self.name=""
        self.bias=""
        self.factuality =""
        self.notes =""
        self.url=""
        self.mbfc = ""

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
    #X_embeddings = gen_or_load_feats("embeddings", h, b, "features/embeddings."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_overlap,X_refuting,X_belief,X_denial,X_doubt,X_fake,X_knowledge,X_negation,X_question,X_report,X_tfidf]
    return X,y

def gen_features_for_prediction (text1,text2):
    t1, t2 = [], []  # t1 is the user selected text and t2 is the body of a collected article
    t1.append(text1)
    t2.append(text2)

    X_overlap = generate_baseline_feats(word_overlap_features, t1, t2, "")
    X_belief = generate_additional_features("expanded_cue_words/belief.txt", t1, t2, "")
    X_denial = generate_additional_features("expanded_cue_words/denial.txt", t1, t2, "")
    X_doubt = generate_additional_features("expanded_cue_words/doubt.txt", t1, t2, "")
    X_fake = generate_additional_features("expanded_cue_words/fake.txt", t1, t2, "")
    X_knowledge = generate_additional_features("expanded_cue_words/knowledge.txt", t1, t2,"")
    X_negation = generate_additional_features("expanded_cue_words/negation.txt", t1, t2,"")
    X_question = generate_additional_features("expanded_cue_words/question.txt", t1, t2,"")
    X_report = generate_additional_features("expanded_cue_words/report.txt", t1, t2, "")
    X_refuting = generate_baseline_feats(refuting_features, t1, t2, "")
    X_polarity = generate_baseline_feats(polarity_features, t1, t2, "")
    X_hand = generate_baseline_feats(hand_features, t1, t2, "")
    X_tfidf = generate_additional_features("tfidf", t1, t2, "")
    #X_embeddings = generate_additional_features("embeddings", t1, t2, "")

    X = np.c_[X_hand, X_polarity, X_overlap, X_refuting, X_belief, X_denial, X_doubt, X_fake, X_knowledge, X_negation, X_question, X_report,X_tfidf]

    return X

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
    limit = 1537  # this is the number of examples the least class "disagree"

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

    d.stances =[]
    d.stances.extend(splitted['unrelated'])
    d.stances.extend(splitted['disagree'])
    d.stances.extend(splitted['agree'])
    d.stances.extend(splitted['discuss'])

    return d


def get_full_dataset():
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    # Load the competition dataset
    competition_dataset = DataSet("competition_test")

    # Merging the train, test to train the chosen model on the full dataset
    d.articles.update(competition_dataset.articles)
    d.stances.extend(competition_dataset.stances)

    return d

def train_and_preserve_model():

    d = get_full_dataset()

    # extracting and vectorizing features
    X, y = generate_features(d.stances,d,"all_data")

    #training the model
    clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
    clf.fit(X, y)

    #preserving the trained model
    dump(clf, 'trained_model.joblib')


def predict_stance(text1,text2):

    X = gen_features_for_prediction(text1,text2)
    clf = load('trained_model.joblib')
    predictions = clf.predict(X)
    predicted_labels = [LABELS[int(p)] for p in predictions]
    return predicted_labels



def get_sources():
    sources = dict()
    with codecs.open("source_credibility.csv", 'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip()
            fields = line.split('\t')
            s = Source()
            s.bias= fields[4]
            s.factuality= fields[5]
            s.mbfc= fields[0]
            s.name= fields[1]
            s.notes=fields[6]
            s.url= fields[7]
            if not s.url == "NIL":
                sources[s.url] = s
    return sources


def return_summary(d,i,sources):
    out ="\n"
    out+="Document #" + str(i) + " says that:"
    for sent in d.most_relevant_sent:
        out+= '\n'+sent
    out+="\nThis document was "
    if d.url != "UNKNOWN":
        out+="\npublished at: " + d.url
    if d.publish_date != "UNKNOWN":
        out+="\npublished on: " + d.publish_date
    for key, source in sources.items():
        if key in d.source or key in d.url:
            out+="\nHere are some info about the source of this document:"
            out+="\nSource name: "+source.name
            out+="\nSource bias:"+source.bias
            if "VERY" in source.factuality: source.factuality= "VERY HIGH"
            out+="\nSource factuality:"+source.factuality
            out+="\nAdditional notes:"+source.notes


    out+='\n'

    return out


def generate_report(relevant_docs,claim,sources):
    out =""
    labels = ["unrelated", "disagree", "agree", "discuss"]
    stance_groups = {label: [] for label in labels}
    for doc in relevant_docs:
        stance_groups[doc.stance].append(doc)

    total_agree = len(stance_groups['agree'])
    total_disagree = len(stance_groups['disagree'])
    summary = ""
    if total_agree > total_disagree:
        out+="The majority of our sources agree with the claim:\n "
        out+=claim
        out+= "\nHere are some:\n"
        i=1
        for d in stance_groups['agree']:
            result = return_summary(d,i,sources)
            summary = summary+result
            i+=1
    elif total_agree < total_disagree:
        out+= "The majority of our sources disagree with the claim:\n "
        out+=claim
        out+= "\nHere are some:\n"
        i=1
        for d in stance_groups['disagree']:
            result = return_summary(d,i,sources)
            summary = summary + result
            i+=1

    out = out + summary
    return out

def fact_check(claim):
    sources = get_sources()
    relevant_docs = do_research(claim)
    # print(len(relevant_docs))
    for doc in relevant_docs:
        predictions = predict_stance(claim, doc.text)
        doc.stance = predictions[0]

    report = generate_report(relevant_docs, claim, sources)

    return report

train_and_preserve_model()
# print(fact_check("Facebook post says that PepsiCo announced Mountain Dew will be discontinued over health concerns."))
