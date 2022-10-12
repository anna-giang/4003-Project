from bagOfWords.BagOfWords import BagOfWords
from wordEmbeddings.WordEmbeddings import WordEmbeddings


print("##### BEGINNING ACCURACY ASSESSMENT #####\n\n")
print("##### Sectioned Advertisement Text #####\n\n")
print("WORD EMBEDDINGS WITH GLOVE\n\n")

diversityInclusion = WordEmbeddings(
    recommendation="diversityInclusion",
    trainingLabelsPath="\\prototype\\labels\\diversityInclusion.csv",
    testLabelsPath="\\prototype\\labels\\diversityInclusionTest.csv",
)

encourageGenders = WordEmbeddings(
    recommendation="encourageGenders",
    trainingLabelsPath="\\prototype\\labels\\encourageGenders.csv",
    testLabelsPath="\\prototype\\labels\\encourageGendersTest.csv",
)

mentionOrgFeatures = WordEmbeddings(
    recommendation="mentionOrgFeatures",
    trainingLabelsPath="\\prototype\\labels\\mentionOrgFeatures.csv",
    testLabelsPath="\\prototype\\labels\\mentionOrgFeaturesTest.csv",
)

print("Diversity Inclusion\n")
diversityInclusion.assessAccuracy()
print("\nEncourage Genders\n")
encourageGenders.assessAccuracy()
print("\nMention Organisation Features\n")
mentionOrgFeatures.assessAccuracy()

print("------------------------------------------------------------------\n\n")

print("BAG OF WORDS\n\n")
diversityInclusion = BagOfWords(
    recommendation="diversityInclusion",
    trainingLabelsPath="\\prototype\\labels\\diversityInclusion.csv",
    testLabelsPath="\\prototype\\labels\\diversityInclusionTest.csv",
)

encourageGenders = BagOfWords(
    recommendation="encourageGenders",
    trainingLabelsPath="\\prototype\\labels\\encourageGenders.csv",
    testLabelsPath="\\prototype\\labels\\encourageGendersTest.csv",
)

mentionOrgFeatures = BagOfWords(
    recommendation="mentionOrgFeatures",
    trainingLabelsPath="\\prototype\\labels\\mentionOrgFeatures.csv",
    testLabelsPath="\\prototype\\labels\\mentionOrgFeaturesTest.csv",
)
print("Diversity Inclusion\n")
diversityInclusion.assessAccuracy()
print("\nEncourage Genders\n")
encourageGenders.assessAccuracy()
print("\nMention Organisation Features\n")
mentionOrgFeatures.assessAccuracy()

print("------------------------------------------------------------------\n")
print("------------------------------------------------------------------\n\n")

print("##### Complete Advertisement Text #####\n\n")
print("WORD EMBEDDINGS WITH GLOVE\n\n")

diversityInclusion = WordEmbeddings(
    recommendation="diversityInclusion",
    trainingLabelsPath="\\prototype\\labels\\diversityInclusion.csv",
    testLabelsPath="\\prototype\\labels\\diversityInclusionTest.csv",
    trainingDataPath="\\prototype\\completeAdText"
)

encourageGenders = WordEmbeddings(
    recommendation="encourageGenders",
    trainingLabelsPath="\\prototype\\labels\\encourageGenders.csv",
    testLabelsPath="\\prototype\\labels\\encourageGendersTest.csv",
)

mentionOrgFeatures = WordEmbeddings(
    recommendation="mentionOrgFeatures",
    trainingLabelsPath="\\prototype\\labels\\mentionOrgFeatures.csv",
    testLabelsPath="\\prototype\\labels\\mentionOrgFeaturesTest.csv",
    trainingDataPath="\\prototype\\completeAdText"
)

print("Diversity Inclusion\n")
diversityInclusion.assessAccuracy()
print("\nEncourage Genders\n")
encourageGenders.assessAccuracy()
print("\nMention Organisation Features\n")
mentionOrgFeatures.assessAccuracy()

print("------------------------------------------------------------------\n\n")

print("BAG OF WORDS\n\n")
diversityInclusion = BagOfWords(
    recommendation="diversityInclusion",
    trainingLabelsPath="\\prototype\\labels\\diversityInclusion.csv",
    testLabelsPath="\\prototype\\labels\\diversityInclusionTest.csv",
    trainingDataPath="\\prototype\\completeAdText"
)

encourageGenders = BagOfWords(
    recommendation="encourageGenders",
    trainingLabelsPath="\\prototype\\labels\\encourageGenders.csv",
    testLabelsPath="\\prototype\\labels\\encourageGendersTest.csv",
    trainingDataPath="\\prototype\\completeAdText"
)

mentionOrgFeatures = BagOfWords(
    recommendation="mentionOrgFeatures",
    trainingLabelsPath="\\prototype\\labels\\mentionOrgFeatures.csv",
    testLabelsPath="\\prototype\\labels\\mentionOrgFeaturesTest.csv",
    trainingDataPath="\\prototype\\completeAdText"
)
print("Diversity Inclusion\n")
diversityInclusion.assessAccuracy()
print("\nEncourage Genders\n")
encourageGenders.assessAccuracy()
print("\nMention Organisation Features\n")
mentionOrgFeatures.assessAccuracy()
