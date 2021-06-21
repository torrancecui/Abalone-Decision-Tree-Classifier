# import numpy as np
import random as rand
import math

low = [0.843, 0.398, 0.389, 0.1498, 0.756, 0.339, 0.181, 0.221]
high = [1.11, 0.545, 0.456, 0.161, 0.982, 0.459, 0.241, 0.3709 ]

class treeNode():
    def __init__(self):
        self.left = None
        self.middle = None
        self.right = None

        self.giniScore = None
        self.nodeData = None

        self.isLeaf = None
        self.classification = None

        self.lowT = None
        self.highT = None
        self.featureIdx = None

    def setTreeNodeData(self, data):
        self.nodeData = data

    def setTreeNodeInfo(self, left, middle, right, giniScore, lowT, highT, featureIdx):
        self.left = left
        self.middle = middle
        self.right = right
        self.giniScore = giniScore
        self.lowT = lowT
        self.highT = highT
        self.featureIdx = featureIdx

    def countClasses(self, data):
        class0, class1, class2 = 0,0,0
        for row in self.nodeData:
            if row[0] == 0:
                class0 += 1
            elif row[0] == 1:
                class1 += 1
            elif row[0] == 2:
                class2 += 1
        return class0, class1, class2

    def isHomo(self, classNum, size):
        homoVal = 0.50

        proportion = float(classNum/size)
        if  proportion >= homoVal:
            return True

    def evaluateHomo(self):
        class0, class1, class2 = self.countClasses(self.nodeData)
        size = len(self.nodeData)
        if self.isHomo(class0, size):
            self.classification = 0
            return True
        elif self.isHomo(class1, size):
            self.classification = 1
            return True
        elif self.isHomo(class2, size):
            self.classification = 2
            return True
        return False

def combine_data_and_labels(data, labels):
    newData = []
    for idx, row in enumerate(data):
        newRow = (labels[idx], row)
        newData.append(newRow)
    return newData

def calculate_indv_gini(split):
    size = float(len(split))

    #check for divide by zero
    if size == 0:
        return 0

    class0, class1, class2 = 0,0,0
    for label in split:
        if label[0] == 0:
            class0 += 1
        if label[0] == 1:
            class1 += 1
        if label[0] == 2:
            class2 += 1

    p0 = float(class0/size)
    p1 = float(class1/size)
    p2 = float(class2/size) 
    
    return 1 - ( (p0 * p0) + (p1 * p1) + (p2 * p2) )


def calculate_gini(left, middle, right):
    left_gini = calculate_indv_gini(left)
    middle_gini = calculate_indv_gini(middle)
    right_gini = calculate_indv_gini(right)

    size = float(len(left)) + float(len(middle)) + float(len(right))

    #check for divide by zero
    if size == 0:
        return 0
    else:
        return (left_gini * float(len(left)) / size) + (middle_gini * float(len(middle)) / size) + (right_gini * float(len(right)) / size)

def test_split(featureIndex, value1, value2, dataset):
    left, middle, right = list(), list(), list()

    if (value1 < value2):
        low = value1
        high = value2
    else:
        low = value2
        high = value1

    for row in dataset:
        if row[1][featureIndex] < low:
            left.append(row)
        elif row[1][featureIndex] >= low and row[1][featureIndex] <= high:
            middle.append(row)
        else:
            right.append(row)
	
    return left, middle, right

def get_best_split(dataset):

    b_feature_index, b_score, b_left, b_middle, b_right  = -1, 2, None, None, None
    
    #for each feature
    for featureIndex in range(8):

        left, middle, right = test_split(featureIndex, low[featureIndex], high[featureIndex], dataset)
        gini = calculate_gini(left, middle, right)

        if gini < b_score:
            b_feature_index, b_score = featureIndex, gini
            b_left = left
            b_middle = middle
            b_right = right

    return b_feature_index, b_score, b_left, b_middle, b_right

def growTree(node):
    # print("Current Node Size: ", len(node.nodeData))
    if node.evaluateHomo():
        # print("REACHED LEAF")
        node.isLeaf = True
        return
    else:
        b_feature_index, b_score, b_left, b_middle, b_right = get_best_split(node.nodeData)
        node.setTreeNodeInfo(b_left, b_middle, b_right, b_score, low[b_feature_index], high[b_feature_index], b_feature_index)

        # print("size of b_left: ", len(b_left))
        # print("size of b_middle: ", len(b_middle))
        # print("size of b_right: ", len(b_right))
        # print("b_score: ", b_score)
        # print("b_feature_index: ", b_feature_index)
        # print("------------------")
        
        node.left = treeNode()
        node.middle = treeNode()
        node.right = treeNode()

        node.left.setTreeNodeData(b_left)
        node.middle.setTreeNodeData(b_middle)
        node.right.setTreeNodeData(b_right)

        if len(node.left.nodeData) == 0 or len(node.middle.nodeData) == 0 or len(node.right.nodeData) == 0:
            # print("REACHED LEAF")
            node.isLeaf = True
            return

        if len(node.left.nodeData) > 0:
            growTree(node.left)
        if len(node.middle.nodeData) > 0:
            growTree(node.middle)
        if len(node.right.nodeData) > 0:
            growTree(node.right)

def predictTesting(treeRoot, testing_data):
    curr = treeRoot
    while not curr.isLeaf:
        featureIdx = curr.featureIdx
        test_featureVal = testing_data[featureIdx]
        if test_featureVal < curr.lowT:
            curr = curr.left
        elif test_featureVal >= curr.lowT and test_featureVal <= curr.highT:
            curr = curr.middle
        else:
            curr = curr.right
    return curr.classification

def run_train_test(training_data, training_labels, testing_data):

    #TODO implement the decision tree and return the prediction

    training_data = combine_data_and_labels(training_data, training_labels)

    # b_feature_index, b_score, b_left, b_right, b_middle = get_best_split(training_data)

    trainTree = treeNode()
    trainTree.setTreeNodeData(training_data)
    growTree(trainTree)

    predOutcomes = []

    for entry in testing_data:
        pred = predictTesting(trainTree, entry)
        predOutcomes.append(pred)

    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[List[float]]
        training_label: List[int]
        testing_data: List[List[float]]

    Output:
        testing_prediction: List[int]
    Example:
    return [1]*len(testing_data)
    """

    # return [1]*len(testing_data)
    return predOutcomes



