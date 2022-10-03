"""
Bag of words prototype script based on Suvansh's code.
"""

import pandas as pd
import numpy as np
import re
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import nltk
nltk.download('stopwords')

CURRENT_PATH = os.getcwd()

# Load text from documents
def loadTextFromFile(directory, filenames, docs):
    trainingDirectory = CURRENT_PATH + "/" + directory
    for filename in os.listdir(trainingDirectory):
        filenames.append(int(filename[:-4]))  # Removes the .txt from the filename
        with open(trainingDirectory + "/" + filename, "r") as file:
            text = file.read()
        docs.append(text)


diversityInclusion = "diversityInclusion"
diversityInclusionFilenames = []
diversityInclusionDocs = []
diversityInclusionTestFilenames = []
diversityInclusionTestDocs = []
loadTextFromFile(
    diversityInclusion + "/" + "training",
    diversityInclusionFilenames,
    diversityInclusionDocs,
)
loadTextFromFile(
    diversityInclusion + "/" + "test",
    diversityInclusionTestFilenames,
    diversityInclusionTestDocs,
)


encourageGenders = "encourageGenders"
encourageGendersFilenames = []
encourageGendersDocs = []
encourageGendersTestFilenames = []
encourageGendersTestDocs = []
loadTextFromFile(
    encourageGenders + "/" + "training",
    encourageGendersFilenames,
    encourageGendersDocs,
)
loadTextFromFile(
    encourageGenders + "/" + "test",
    encourageGendersTestFilenames,
    encourageGendersTestDocs,
)

mentionOrgFeatures = "mentionOrgFeatures"
mentionOrgFeaturesFilenames = []
mentionOrgFeaturesDocs = []
mentionOrgFeaturesTestFilenames = []
mentionOrgFeaturesTestDocs = []
loadTextFromFile(
    mentionOrgFeatures + "/" + "training",
    mentionOrgFeaturesFilenames,
    mentionOrgFeaturesDocs,
)
loadTextFromFile(
    mentionOrgFeatures + "/" + "test",
    mentionOrgFeaturesTestFilenames,
    mentionOrgFeaturesTestDocs,
)

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Preparing dataframes

# Builds a table with two columns: filename ('id'), and the text from the file (the job ad)

diversityInclusionData = pd.DataFrame()
diversityInclusionData["id"] = diversityInclusionFilenames
diversityInclusionData["text"] = diversityInclusionDocs
diversityInclusionTestData = pd.DataFrame()
diversityInclusionTestData["id"] = diversityInclusionTestFilenames
diversityInclusionTestData["text"] = diversityInclusionTestDocs

encourageGendersData = pd.DataFrame()
encourageGendersData["id"] = encourageGendersFilenames
encourageGendersData["text"] = encourageGendersDocs
encourageGendersTestData = pd.DataFrame()
encourageGendersTestData["id"] = encourageGendersTestFilenames
encourageGendersTestData["text"] = encourageGendersTestDocs

mentionOrgFeaturesData = pd.DataFrame()
mentionOrgFeaturesData["id"] = mentionOrgFeaturesFilenames
mentionOrgFeaturesData["text"] = mentionOrgFeaturesDocs
mentionOrgFeaturesTestData = pd.DataFrame()
mentionOrgFeaturesTestData["id"] = mentionOrgFeaturesTestFilenames
mentionOrgFeaturesTestData["text"] = mentionOrgFeaturesTestDocs

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
diversityInclusionTestData["text"] = diversityInclusionTestData["text"].apply(
    lambda x: clean_text(x)
)

encourageGendersTestData["text"] = encourageGendersData["text"].apply(
    lambda x: clean_text(x)
)
encourageGendersTestData["text"] = encourageGendersTestData["text"].apply(
    lambda x: clean_text(x)
)

mentionOrgFeaturesData["text"] = mentionOrgFeaturesData["text"].apply(
    lambda x: clean_text(x)
)
mentionOrgFeaturesTestData["text"] = mentionOrgFeaturesTestData["text"].apply(
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
diversityInclusionTestData["text"] = diversityInclusionTestData["text"].apply(
    lambda x: remove_stopwords(x)
)

encourageGendersData["text"] = encourageGendersData["text"].apply(
    lambda x: remove_stopwords(x)
)
encourageGendersTestData["text"] = encourageGendersTestData["text"].apply(
    lambda x: remove_stopwords(x)
)

mentionOrgFeaturesData["text"] = mentionOrgFeaturesData["text"].apply(
    lambda x: remove_stopwords(x)
)
mentionOrgFeaturesTestData["text"] = mentionOrgFeaturesTestData["text"].apply(
    lambda x: remove_stopwords(x)
)

# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Reading pre-labelled target classes

diversityInclusionLabels = pd.read_csv(
    CURRENT_PATH + "/labels/diversityInclusion.csv"
)
diversityInclusionData = pd.merge(diversityInclusionData, diversityInclusionLabels)
diversityInclusionTestLabels = pd.read_csv(
    CURRENT_PATH + "/labels/diversityInclusionTest.csv"
)
diversityInclusionTestData = pd.merge(
    diversityInclusionTestData, diversityInclusionTestLabels
)


encourageGendersLabels = pd.read_csv(CURRENT_PATH + "/labels/encourageGenders.csv")
encourageGendersData = pd.merge(encourageGendersData, encourageGendersLabels)
encourageGendersTestLabels = pd.read_csv(
    CURRENT_PATH + "/labels/encourageGendersTest.csv"
)
encourageGendersTestData = pd.merge(
    encourageGendersTestData, encourageGendersTestLabels
)


mentionOrgFeaturesLabels = pd.read_csv(
    CURRENT_PATH + "/labels/mentionOrgFeatures.csv"
)
mentionOrgFeaturesData = pd.merge(mentionOrgFeaturesData, mentionOrgFeaturesLabels)
mentionOrgFeaturesTestLabels = pd.read_csv(
    CURRENT_PATH + "/labels/mentionOrgFeaturesTest.csv"
)
mentionOrgFeaturesTestData = pd.merge(
    mentionOrgFeaturesTestData, mentionOrgFeaturesTestLabels
)


# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Splitting data & implementing bag of words
diversityInclusionBag = CountVectorizer(ngram_range=(2, 2))
diversityInclusionText = diversityInclusionBag.fit_transform(
    diversityInclusionData["text"]
)
diversityInclusionLabels = np.array(diversityInclusionData[diversityInclusion])
diversityInclusionTextFeatures = diversityInclusionText.toarray()


diversityInclusionTestBag = CountVectorizer(
    ngram_range=(2, 2),
    vocabulary=diversityInclusionBag.vocabulary_,  # Create a CountVectorizer ONLY considering the vocabulary of the training data.
)
diversityInclusionTestText = diversityInclusionTestBag.fit_transform(
    diversityInclusionTestData["text"]
)
diversityInclusionTestLabels = np.array(diversityInclusionTestData[diversityInclusion])
diversityInclusionTestTextFeatures = diversityInclusionTestText.toarray()

""""""

encourageGendersBag = CountVectorizer(ngram_range=(2, 2))
encourageGendersText = encourageGendersBag.fit_transform(encourageGendersData["text"])
encourageGendersLabels = np.array(encourageGendersData[encourageGenders])
encourageGendersTextFeatures = encourageGendersText.toarray()

encourageGendersTestBag = CountVectorizer(
    ngram_range=(2, 2), vocabulary=encourageGendersBag.vocabulary_
)
encourageGendersTestText = encourageGendersTestBag.fit_transform(
    encourageGendersTestData["text"]
)
encourageGendersTestLabels = np.array(encourageGendersTestData[encourageGenders])
encourageGendersTestTextFeatures = encourageGendersTestText.toarray()

""""""

mentionOrgFeaturesBag = CountVectorizer(ngram_range=(2, 2))
mentionOrgFeaturesText = mentionOrgFeaturesBag.fit_transform(
    mentionOrgFeaturesData["text"]
)
mentionOrgFeaturesLabels = np.array(mentionOrgFeaturesData[mentionOrgFeatures])
mentionOrgFeaturesTextFeatures = mentionOrgFeaturesText.toarray()

mentionOrgFeaturesTestBag = CountVectorizer(
    ngram_range=(2, 2), vocabulary=mentionOrgFeaturesBag.vocabulary_
)
mentionOrgFeaturesTestText = mentionOrgFeaturesTestBag.fit_transform(
    mentionOrgFeaturesTestData["text"]
)
mentionOrgFeaturesTestLabels = np.array(mentionOrgFeaturesTestData[mentionOrgFeatures])
mentionOrgFeaturesTestTextFeatures = mentionOrgFeaturesTestText.toarray()

""""""

trainingData = [
    [diversityInclusionTextFeatures, diversityInclusionLabels],
    [encourageGendersTextFeatures, encourageGendersLabels],
    [mentionOrgFeaturesTextFeatures, mentionOrgFeaturesLabels],
]

# Creating ML model (random forest classifier)
models = []

for i in range(len(trainingData)):
    rf = RandomForestClassifier(n_estimators=200)
    models.append(rf.fit(trainingData[i][0], trainingData[i][1]))

predictions = []
predictions.append(models[0].predict(diversityInclusionTestTextFeatures))
predictions.append(
    models[1].predict(encourageGendersTestTextFeatures)
)  # Encourage both genders
predictions.append(
    models[2].predict(mentionOrgFeaturesTestTextFeatures)
)  # Mention of work features

# Evaluating precisions
print("\t Classification report for", diversityInclusion, "\n")
print(
    metrics.classification_report(
        diversityInclusionTestLabels, predictions[0], digits=5
    )
)
print("-------------------------------------------------------------------------")

print("\t Classification report for", encourageGenders, "\n")
print(
    metrics.classification_report(encourageGendersTestLabels, predictions[1], digits=5)
)
print("-------------------------------------------------------------------------")

print("\t Classification report for", mentionOrgFeatures, "\n")
print(
    metrics.classification_report(
        mentionOrgFeaturesTestLabels, predictions[2], digits=5
    )
)
