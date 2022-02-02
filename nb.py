import numpy as np

def helper_count(array,element):
    count = 0
    for eachData in array:
        if eachData == element:
            count+=1
    return count


def predict(theta,pi,new_sample,classes,vocab):
    probs = {}
    # print(classes)
    for cls in classes:
        class_prob = np.log(pi[cls])
        for i in range(0, len(new_sample)):
            if new_sample[i] not in vocab:
                continue
            relative_values = theta[cls][new_sample[i]]
            class_prob += np.log(relative_values)
        probs[cls] = class_prob
    return probs




def vocabulary(data):
    """
    Creates the vocabulary from the data.
    :param data: List of lists, every list inside it contains words in that sentence.
                 len(data) is the number of examples in the data.
    :return: Set of words in the data

    """
    # sorted(data)
    vocab = set()
    for eachData in data:
        for each in eachData:
            vocab.add(each)
    return vocab

def train(train_data, train_labels, vocab):
    """
    Estimates the probability of a specific word given class label using additive smoothing with smoothing constant 1.
    :param train_data: List of lists, every list inside it contains words in that sentence.
                       len(train_data) is the number of examples in the training data.
    :param train_labels: List of class names. len(train_labels) is the number of examples in the training data.
    :param vocab: Set of words in the training set.
    :return: theta, pi. theta is a dictionary of dictionaries. At the first level, the keys are the class names. At the
             second level, the keys are all of the words in vocab and the values are their estimated probabilities.
             pi is a dictionary. Its keys are class names and values are their probabilities.
    """
    pi = dict()
    theta = dict()
    uniqueLabels = list(dict.fromkeys(train_labels))
    for unique in uniqueLabels:
        count = 0
        for labels in train_labels:
            if labels == unique:
                count+=1
        pi[unique] = count/len(train_labels)

    indices = dict()

    for unique in uniqueLabels:
        index = 0
        indices[unique] = []
        for labels in train_labels:
            if labels == unique:
                indices[unique].append(index)
            index+=1

    # print(uniqueLabels)
    #
    # print(indices)
    #
    # exit()
    count = dict()
    for unique in uniqueLabels:
        temp = []
        for eachIndex in indices[unique]:
            temp.append(train_data[eachIndex])
        new_temp = []
        for eachTemp in temp:
            for each in eachTemp:
                new_temp.append(each)

        count[unique] = new_temp

    # print(count)
    # print(len(count))
    countDict = dict()


    for unique in uniqueLabels:
        count_for_dict = dict()
        flag = 0
        for eachVocab in vocab:
            countForThis = helper_count(count[unique],eachVocab)
            if countForThis == 0:
                # countForThis = 1
                flag+=1
            count_for_dict[eachVocab] = countForThis
        new_count_for_dict = dict()
        for c in count_for_dict:
            new_count_for_dict[c] = count_for_dict[c] + 1

        countDict[unique] = new_count_for_dict
    # print(countDict)
    # exit()
    for unique in uniqueLabels:
        total_number = 0
        for each in vocab:
            total_number += countDict[unique][each]
        # print(total_number)
        # print(countDict[unique])
        new_dict = dict()
        for eachRecord in countDict[unique]:
            # print(eachRecord)
            res = countDict[unique][eachRecord]
            # print(res/total_number)
            #theta[unique][eachRecord] = res/total_number
            new_dict[eachRecord] = res/total_number
        theta[unique] = new_dict

    #print(theta)
    return theta, pi




def test(theta, pi, vocab, test_data):
    """
    Calculates the scores of a test data given a class for each class. Skips the words that are not occurring in the
    vocabulary.
    :param theta: A dictionary of dictionaries. At the first level, the keys are the class names. At the second level,
                  the keys are all of the words in vocab and the values are their estimated probabilities.
    :param pi: A dictionary. Its keys are class names and values are their probabilities.
    :param vocab: Set of words in the training set.
    :param test_data: List of lists, every list inside it contains words in that sentence.
                      len(test_data) is the number of examples in the test data.
    :return: scores, list of lists. len(scores) is the number of examples in the test set. Every inner list contains
             tuples where the first element is the score and the second element is the class name.
    """


    classes = list(pi.keys())
    result = []
    for eachTest in test_data:
        probs = predict(theta,pi,eachTest,classes,vocab)
        valuesProbs = list(probs.values())
        keysProbs = list(probs.keys())
        index = 0
        eachResult = []
        for value in valuesProbs:
            eachResult.append((value,keysProbs[index]))
            index+=1
        result.append(eachResult)
    return result


def extractDataset(train_filename,train_label_filename,test_filename,test_label_filename):

    train_data = []
    test_data = []
    train_label_data = []
    test_label_data = []

    f = open(train_filename)
    for eachLine in f:
        train_data.append(eachLine.split())
    f.close()
    f = open(train_label_filename)
    for eachLine in f:
        withoutNewLine = eachLine.strip('\n')
        train_label_data.append(withoutNewLine)
    f.close()
    f = open(test_filename)
    for eachLine in f:
        test_data.append(eachLine.split())
    f.close()
    f = open(test_label_filename)
    for eachLine in f:
        withoutNewLine = eachLine.strip('\n')
        test_label_data.append(withoutNewLine)
    f.close()
    return train_data,train_label_data,test_data,test_label_data

def argMax(probs):
    result = []
    for eachProbs in probs:
        new_values_list = []
        for each in eachProbs:
            new_values_list.append(each[0])
        arg_max = max(new_values_list)
        index = 0
        for values in new_values_list:
            if values == arg_max:
                break
            index+=1
        result.append(eachProbs[index][1])

    return result









if __name__ == "__main__":

    train_data, train_label_data, test_data, test_label_data = extractDataset("./hw4_data/news/train_data.txt","./hw4_data/news/train_labels.txt","./hw4_data/news/test_data.txt","./hw4_data/news/test_labels.txt")

    vocab = vocabulary(train_data)
    new_train_data = []
    for eachData in train_data:
        new_train_row = []
        for each in eachData:
            resultString = ""
            for character in each:
                if character.isalnum():
                    resultString += character
            if resultString == "":
                continue
            new_train_row.append(resultString)
        new_train_data.append(new_train_row)
    new_test_data = []
    for eachData in test_data:
        new_test_row = []
        for each in eachData:
            resultString = ""
            for character in each:
                if character.isalnum():
                    resultString+=character
            if resultString == "":
                continue
            new_test_row.append(resultString)
        new_test_data.append(new_test_row)
    theta , pi = train(new_train_data,train_label_data,vocab)
    # print(theta)
    # print(pi)
    probs = test(theta,pi,vocab,new_test_data)
    # print(probs)
    predictedValues = argMax(probs)
    # print(predictedValues)


    ### accuracy
    index=0
    countTrueLabels = 0
    for test_label in test_label_data:
        if predictedValues[index] == test_label:
            countTrueLabels+=1
        index+=1

    print("Naive Bayes Test Accuracy %",100*countTrueLabels/len(test_label_data))
