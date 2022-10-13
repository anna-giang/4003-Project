"""
Example of how to use the BagOfWords class to integrate with the rest of the 
application.
"""

from bagOfWords.BagOfWords import BagOfWords

# Initialise a model trained using split advertisement text for each of the recommendations
diversityInclusion = BagOfWords(
    recommendation="diversityInclusion",
    trainingLabelsPath="/prototype/labels/diversityInclusion.csv",
    testLabelsPath="/prototype/labels/diversityInclusionTest.csv",
)

encourageGenders = BagOfWords(
    recommendation="encourageGenders",
    trainingLabelsPath="/prototype/labels/encourageGenders.csv",
    testLabelsPath="/prototype/labels/encourageGendersTest.csv",
)

mentionOrgFeatures = BagOfWords(
    recommendation="mentionOrgFeatures",
    trainingLabelsPath="/prototype/labels/mentionOrgFeatures.csv",
    testLabelsPath="/prototype/labels/mentionOrgFeaturesTest.csv",
)

# The text from the job ad you want to analyse
jobAd = """YOUR_JOB_AD_HERE!"""

# Call .predict() on the job ad text, which returns 1/0 (true/false) whether the recommendation is met.
print("###### PREDICTIONS ######")
print("Diversity and Inclusion: " + str(diversityInclusion.predict(jobAd)))
print("Encourage genders: " + str(encourageGenders.predict(jobAd)))
print("Mention organisation features: " +
      str(mentionOrgFeatures.predict(jobAd)))
print()

# Call .assessAccuracy() print the accuracy results
print("###### ACCURACIES ######")
print("Diversity Inclusion")
diversityInclusion.assessAccuracy()
print("\nEncourage Genders")
encourageGenders.assessAccuracy()
print("\nMention Organisation Features")
mentionOrgFeatures.assessAccuracy()
