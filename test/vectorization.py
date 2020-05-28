import pandas as pd
# import re
from sklearn.feature_extraction.text import TfidfVectorizer

# the term frequency tf(t,d) is the number of times that term t occurs in document d
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bagOfWordsCount)
    return tfDict

# The inverse document frequency is a measure of how much information the word provides
def computeIDF(documents):
    from math import log
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if(val > 0):
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = log(N/float(val))
    return idfDict

# tfidf = tf(t,d)*idf(t,D)
def computeTFIDF(tfbagOfWords, idfs):
    tfidf = {}
    for word, val in tfbagOfWords.items():
        tfidf[word] = val*idfs[word]
    return tfidf

if __name__ == "__main__":
    doc = "Tôi tên là A. Tôi 20 tuổi, sinh sống ở Hải Dương. Tôi học Đại học Công Nghệ."
    doc2 = "Chào bạn, tôi là B. Tôi ở Hà Nội và học cùng trường với bạn."
    doc = str(doc).lower()
    doc2 = str(doc2).lower()
    # doc = re.sub(r'[^ a-z]', '', doc) # removing special characters
    # doc2 = re.sub(r'[^ a-z]', '', doc2) # removing special characters
    bagOfWordDoc = doc.split(' ')
    bagOfWordDoc2 = doc2.split(' ')

    uniqueWords = set(bagOfWordDoc).union(set(bagOfWordDoc2))
    # print(bagOfWordDoc)
    numOfWords = dict.fromkeys(uniqueWords, 0)
    numOfWords2 = dict.fromkeys(uniqueWords, 0)

    for word in bagOfWordDoc:
        numOfWords[word] += 1
    for word in bagOfWordDoc2:
        numOfWords2[word] += 1
    # print(numOfWords)
    tf = computeTF(numOfWords, bagOfWordDoc)
    tf2 = computeTF(numOfWords2, bagOfWordDoc2)
    
    idfs = computeIDF([numOfWords, numOfWords2])
    
    ifidf = computeTFIDF(tf, idfs)
    ifidf2 = computeTFIDF(tf2, idfs)
    # df = pd.DataFrame([ifidf, ifidf2])
    # print(df)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc, doc2])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    print(df)