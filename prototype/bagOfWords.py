import pandas as pd
import numpy as np
import re
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

CURRENT_PATH = os.getcwd()

# Load text from documents
def loadTextFromFile(directory, filenames, docs):
    trainingDirectory = CURRENT_PATH + "\\" + directory
    for filename in os.listdir(trainingDirectory):
        filenames.append(int(filename[:-4]))  # Removes the .txt from the filename
        with open(trainingDirectory + "\\" + filename, "r") as file:
            text = file.read()
        docs.append(text)


diversityInclusion = "diversityInclusion"
diversityInclusionFilenames = []
diversityInclusionDocs = []
loadTextFromFile(
    diversityInclusion, diversityInclusionFilenames, diversityInclusionDocs
)


encourageGenders = "encourageGenders"
encourageGendersFilenames = []
encourageGendersDocs = []
loadTextFromFile(encourageGenders, encourageGendersFilenames, encourageGendersDocs)

mentionOrgFeatures = "mentionOrgFeatures"
mentionOrgFeaturesFilenames = []
mentionOrgFeaturesDocs = []
loadTextFromFile(
    mentionOrgFeatures, mentionOrgFeaturesFilenames, mentionOrgFeaturesDocs
)

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Preparing dataframes

# Builds a table with two columns: filename ('id'), and the text from the file (the job ad)

diversityInclusionData = pd.DataFrame()
diversityInclusionData["id"] = diversityInclusionFilenames
diversityInclusionData["text"] = diversityInclusionDocs

encourageGendersData = pd.DataFrame()
encourageGendersData["id"] = encourageGendersFilenames
encourageGendersData["text"] = encourageGendersDocs

mentionOrgFeaturesData = pd.DataFrame()
mentionOrgFeaturesData["id"] = mentionOrgFeaturesFilenames
mentionOrgFeaturesData["text"] = mentionOrgFeaturesDocs

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Cleaning text
def clean_text(text):
    # Remove punctutation
    text = re.sub("[^a-zA-Z]", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Convert to lower
    text = text.lower()
    # Remove whitespaces
    text = " ".join(text.split())
    return text


diversityInclusionData["text"] = diversityInclusionData["text"].apply(
    lambda x: clean_text(x)
)

encourageGendersData["text"] = diversityInclusionData["text"].apply(
    lambda x: clean_text(x)
)

mentionOrgFeaturesData["text"] = mentionOrgFeaturesData["text"].apply(
    lambda x: clean_text(x)
)

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Removing stopwords

stop = set(stopwords.words("english"))

# A function to remove stopwords and short length words (< 2)
def remove_stopwords(text):
    new = []
    for word in text.split():
        if word not in stop and len(word) > 1:
            new.append(word)

    return " ".join(new)


diversityInclusionData["text"] = diversityInclusionData["text"].apply(
    lambda x: remove_stopwords(x)
)

encourageGendersData["text"] = encourageGendersData["text"].apply(
    lambda x: remove_stopwords(x)
)

mentionOrgFeaturesData["text"] = mentionOrgFeaturesData["text"].apply(
    lambda x: remove_stopwords(x)
)

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Reading pre-labelled target classes

diversityInclusionLabels = pd.read_csv(
    CURRENT_PATH + "\\labels\\diversityInclusion.csv"
)
diversityInclusionData = pd.merge(diversityInclusionData, diversityInclusionLabels)

encourageGendersLabels = pd.read_csv(CURRENT_PATH + "\\labels\\encourageGenders.csv")
encourageGendersData = pd.merge(encourageGendersData, encourageGendersLabels)

mentionOrgFeaturesLabels = pd.read_csv(
    CURRENT_PATH + "\\labels\\mentionOrgFeatures.csv"
)
mentionOrgFeaturesData = pd.merge(mentionOrgFeaturesData, mentionOrgFeaturesLabels)


# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Splitting data & implementing bag of words
diversityInclusionBag = CountVectorizer(ngram_range=(2, 2))
diversityInclusionText = diversityInclusionBag.fit_transform(
    diversityInclusionData["text"]
)
diversityInclusionLabels = np.array(diversityInclusionData[diversityInclusion])
diversityInclusionTextFeatures = diversityInclusionText.toarray()
deXTrain, deXtest, deYtrain, deYtest = train_test_split(
    diversityInclusionTextFeatures,
    diversityInclusionLabels,
    test_size=0.2,
    random_state=200,
)

encourageGendersBag = CountVectorizer(ngram_range=(2, 2))
encourageGendersText = encourageGendersBag.fit_transform(encourageGendersData["text"])
encourageGendersLabels = np.array(encourageGendersData[encourageGenders])
encourageGendersTextFeatures = encourageGendersText.toarray()
egXtrain, egXtest, egYtrain, egYtest = train_test_split(
    encourageGendersTextFeatures,
    encourageGendersLabels,
    test_size=0.2,
    random_state=200,
)

mentionOrgFeaturesBag = CountVectorizer(ngram_range=(2, 2))
mentionOrgFeaturesText = mentionOrgFeaturesBag.fit_transform(
    mentionOrgFeaturesData["text"]
)
mentionOrgFeaturesLabels = np.array(mentionOrgFeaturesData[mentionOrgFeatures])
mentionOrgFeaturesTextFeatures = mentionOrgFeaturesText.toarray()
mofXtrain, mofXtest, mofYtrain, mofYtest = train_test_split(
    mentionOrgFeaturesTextFeatures,
    mentionOrgFeaturesLabels,
    test_size=0.2,
    random_state=200,
)


trainingData = [[deXTrain, deYtrain], [egXtrain, egYtrain], [mofXtrain, mofYtrain]]

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Creating ML model (random forest classifier)
models = []

for i in range(len(trainingData)):
    rf = RandomForestClassifier(n_estimators=200)
    models.append(rf.fit(trainingData[i][0], trainingData[i][1]))

predictions = []
predictions.append(models[0].predict(deXtest))  # D&E
predictions.append(models[1].predict(egXtest))  # Encourage both genders
predictions.append(models[2].predict(mofXtest))  # Mention of work features

# Evaluating precisions
print("\t Classification report for", diversityInclusion, "\n")
print(metrics.classification_report(deYtest, predictions[0], digits=5))
print("-------------------------------------------------------------------------")

print("\t Classification report for", encourageGenders, "\n")
print(metrics.classification_report(egYtest, predictions[1], digits=5))
print("-------------------------------------------------------------------------")

print("\t Classification report for", mentionOrgFeatures, "\n")
print(metrics.classification_report(mofYtest, predictions[2], digits=5))
