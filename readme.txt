
1. To understand the Fake News Challenge-1 baseline codes, read the file README-baseline.md


To run the scripts of our model training, testing , and prediciton: you need to have Python3 >3.6 installed, along with the following packages:
you can install all of them using pip for python3, using the command:
pip3 install packageName
tqdm
sklearn
numpy
scipy
nltk
google
torch
pytorch_pretrained_bert
gensim
newspaper3k
you also need to install "wordnet" and "punkt" packages from nltk, type:
python3
inside the python environemtn, type:
>>import nltk
>>nltk.download("punkt")
>>nltk.download("wordnet")

2. To train the new model for production (train it on the full data set and preserve it)
uncomment line 248 (train_and_preserve_model()) from the file "main.py" and run the script:
python3 main.py
The trained model will be preserved in a file nemd "trained_model.joblib" in the same directory. In addition to that, the tfidf dictionary built from the training dataset is also preserved to save time in the prediction step (the file is tfidf_vectorizer.pickle"
3. To fact-check a claim using the preserved model, uncomment the last line of the "main.py" file (line 249) and run main.py
python3 main.py
NOTE: make sure to have "trained_model.joblib" in the same directrory level with main.py

4. To train the new model for experiment purposes (train on the train data set and test on the testing data set):
run the file fnc_kfold_original.py:
python3 fnc_kfold_original.py

to remove features from the model, go to the method named "generate_features", comment the line of that feature (from line23 to line 35, then remove the entry of the feature from the array X in line 38.

to add features, uncomment the feature line, and add the entry (the feature name) to the array X in line 36.
NOTE: claculating features for the first time takes time, but features are all preserved in the folder "features" after the first run. In the second run, the precalculated features are loaded from those files to train the model.
5. The script used to expand the cue lists by the most similar 10 words is "expand_cue_list.py", the expanded cue lists can be found in the directory "expanded_cue_words". You don't need to do anything here as the lists are already generated and used in the fnc_kfold_original.py" file, this is just informative step.

6. source credibility information are in "source_credibility.csv". This file is used in the report genration method in "main.py"

7. The document collector module is the file "google_search", it is part of the fact-checking method and is called when you run "main.py" to fact-check an information.

8. To train the model on a balanced dataset, run the file fnc_kfold_sklrn.py:
python3 fnc_kfold_sklrn.py


If you encounter any problem, please feel free to contact me at israa.jaradat@mavs.uta.edu
Thank you

