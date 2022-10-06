import string
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import os
import re
from nltk.corpus import stopwords


class BagOfWords:
    def __init__(
        self, recommendation: string, trainingLabelsPath: string, testLabelsPath=None
    ):

        self.currentPath = os.getcwd()
        self.recommendation = recommendation
        self.trainingLabelsPath = trainingLabelsPath
        self.testLabelsPath = testLabelsPath
        self.vocabulary = None
        self.model = None
        self.train()

    def train(self) -> None:
        trainingDataDirectory = "prototype/" + self.recommendation + "/" + "training"
        pathToTrainingLabelsCsv = self.currentPath + self.trainingLabelsPath

        data = self.__prepareData(trainingDataDirectory, pathToTrainingLabelsCsv)
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        fittedText = vectorizer.fit_transform(data["text"])
        textFeatures = fittedText.toarray()

        self.vocabulary = vectorizer.vocabulary_

        labels = np.array(data[self.recommendation])
        rf = RandomForestClassifier(n_estimators=200)
        self.model = rf.fit(textFeatures, labels)

    def predict(self, sample):
        if not self.model:
            self.train()
        dataFrame = pd.DataFrame()
        dataFrame["id"] = ["advertisement"]
        dataFrame["text"] = [sample]
        dataFrame["text"] = (
            dataFrame["text"]
            .apply(lambda x: Util.clean_text(x))
            .apply(lambda x: Util.remove_stopwords(x))
        )
        features = self.__vectorizeSample(dataFrame)
        return self.model.predict(features)[0]

    def assessAccuracy(self) -> None:
        """
        Prints the classification report of the accuracy of the model based on test data.
        """
        if not self.model:
            self.train()

        if not self.testLabelsPath:
            print("Please set path to test labels.")
            return

        testDataDirectory = "prototype/" + self.recommendation + "/test"
        pathToTestLabelsCsv = self.currentPath + self.testLabelsPath
        testData = self.__prepareData(testDataDirectory, pathToTestLabelsCsv)

        testTextFeatures = self.__vectorizeSample(testData)
        testLabels = np.array(testData[self.recommendation])
        print(
            metrics.classification_report(
                testLabels,
                self.model.predict(testTextFeatures),
                digits=5,
            )
        )

    def __vectorizeSample(self, data: pd.DataFrame) -> np.ndarray:
        vectorizer = CountVectorizer(ngram_range=(2, 2), vocabulary=self.vocabulary)
        fittedText = vectorizer.fit_transform(data["text"])
        textFeatures = fittedText.toarray()
        return textFeatures

    def __prepareData(self, dataDirectory, pathTolabelsCsv) -> pd.DataFrame:
        filenames = []
        docs = []

        Util.loadTextFromFile(
            dataDirectory,
            filenames,
            docs,
        )

        dataFrame = pd.DataFrame()
        dataFrame["id"] = filenames
        dataFrame["text"] = docs
        dataFrame["text"] = (
            dataFrame["text"]
            .apply(lambda x: Util.clean_text(x))
            .apply(lambda x: Util.remove_stopwords(x))
        )
        labels = pd.read_csv(
            pathTolabelsCsv, converters={"id": str}
        )  # Read 'id' column as a string
        dataFrame = pd.merge(dataFrame, labels)
        return dataFrame


class Util:
    currentPath = os.getcwd()

    @staticmethod
    def loadTextFromFile(directory, filenames, docs):
        trainingDirectory = Util.currentPath + "/" + directory
        for filename in os.listdir(trainingDirectory):
            filenames.append(filename[:-4])  # Removes the .txt from the filename
            with open(trainingDirectory + "/" + filename, "r") as file:
                text = file.read()
            docs.append(text)

    @staticmethod
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

    @staticmethod
    def remove_stopwords(text: string):
        """
        Removes stopwords and short length words (< 2)
        """
        stop = set(stopwords.words("english"))
        new = []
        for word in text.split():
            if word not in stop and len(word) > 1:
                new.append(word)

        return " ".join(new)
