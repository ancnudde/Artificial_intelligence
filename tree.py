import numpy as np
from collections import namedtuple, Counter, OrderedDict
from sklearn import datasets


class CostFunction():
    """
    A class containing the cost functions used to
    estimate the best splits in the tree.
    """

    def gini(split_labels):
        """
        Computes the gini score for the given data subset, using their labels
        ('split_labels' parameter).

        Gini score is given by the following formula:

            G = Sum(P_k^2);
            With k being the probability of an item being of class k.
        """
        classes = set(split_labels)
        class_count = Counter(split_labels)
        gini_vector = [class_count[y] / len(split_labels) for y in classes]
        gini_score = sum(class_value ** 2 for class_value in gini_vector)
        return (gini_score)


class Node():
    """
    A helper class defining the nodes of the DecisionTree class.

    Each Node object contains a subset of the data along with a summary of the
    corresponding labels, the informations concerning the split point of this
    data subset, and and boolean telling whether the node is pure or not.

    The node is linked to it's left and right children nodes, that are also
    Node objects.

    The node is also assigned a value that corresponds to the most represented
    class in it, allowing predictions.

    If the node is a leaf, this value is showed when the node is printed. If
    it is not a leaf, the split informations and the labels summary are showed
    instead.
    """

    def __init__(self, data, node_value):
        # Datasubset of the node.
        self.data = data
        # Summary of the labels in the node.
        self.labels = None
        # Links to the children nodes.
        self.left_child = None
        self.right_child = None
        # Most frequent class in the node.
        self.node_value = node_value
        self.split = int(node_value)
        # Purity of the node.
        # A node is pure if all items in it are of the same class.
        self.ispure = False
        self.children = []

    def __str__(self):
        """
        Prints the split informations and the summary of the labels.
        If the root is a leaf, the node value is printed instead of the split.
        """
        return "{}, Classes: {}".format(self.split, self.labels)

    def __repr__(self):
        """
        Prints the split informations and the summary of the labels.
        If the root is a leaf, the node value is printed instead of the split.
        """
        return "{}, Classes: {}".format(self.split, self.labels)

    def is_pure(self, labels):
        """
        Defines whether a node is pure or not. A node is said to be pure when
        all the  samples in it are of the same class. 

        Uses the 'labels' parameter, an array containing the label, to check
        for the number of classes in the node.
        """
        # Checks if there is only one class.
        purity = True if len(set(labels)) == 1 else False
        return purity


class DecisionTree():
    """
    Decision tree class.

    This class generates the decision tree model composed of nodes objects.

    The DecisionTree object contains the dataset and a cost function, and a max
    depth parameter, defining the maximum number of consecutive splits in a
    branch. It builds the tree by calling recursively Node objects with
    splitted subsets of the data.

    The cost function is used the determine the best split. The best split is
    defined among all possible ones by selecting the split point giving the
    highest score. By default, the cost function is the Gini score.

    The max_depth parameter defines how deep is the tree, and therefore it's
    complexity. A tree that is too deep is more likely to overfit due to very
    specific splits, while a tree that is too shallow is more likely to 
    underfit due to the lack of complexity.
    """

    def __init__(self, data, labels, cost_function, max_depth):
        # Training dataset.
        self.data = data
        self.labels = labels
        # Function used to evaluate the splits.
        self.cost = cost_function
        # Maximum depth allowed for the tree.
        self.max_depth = max_depth
        # Construction of the tree.
        self.root = self.extend_tree(data, labels, max_depth)

    def __repr__(self):
        return self.pprint_tree(self.root)

    def split_data(self, feature, split_value):
        """
        Splits the feature data in two subsets using split_value as pivot.
        """
        # Gets indices of data that have feature lower than the pivot.
        left = np.where(feature < split_value)[0]
        # Gets indices of data that have feature higher than the pivot.
        right = np.where(feature >= split_value)[0]
        return (left, right)

    def label_split(self, labels, left_subset, right_subset):
        """
        Extracts the labels corresponding to each subset.
        """
        # Extracts the labels of the data having indices found with split_data.
        left_labels = np.array([labels[index] for index in left_subset])
        right_labels = np.array([labels[index] for index in right_subset])
        return (left_labels, right_labels)

    def compute_score(self, left_labels, right_labels):
        """
        Use the labels of the two subsets to compute the score for the split.

        The score correspond to the sum of the scores for the two subsets
        normalized by the proportion of the values that are in these subsets
        compared to the total data.
        """
        # Total data length.
        n = len(left_labels) + len(right_labels)
        # Score for left subset.
        left_score = len(left_labels) * self.cost(left_labels) / n
        # Score for right subset.
        right_score = len(right_labels) * self.cost(right_labels) / n
        # Returns the total score.
        return (left_score + right_score)

    def find_split(self, features, labels):
        """
        Find the best possible split for the given node.
        """
        # Initializes the container for split point informations.
        SplitPoint = namedtuple('Split', ['feature', 'value', 'score'])
        best_split = SplitPoint(feature=None, value=None, score=-1)
        for i, vector in enumerate(features, 0):
            sorted_values = np.sort(list(set(vector)))
            for j, data_point in enumerate(sorted_values[1:], 1):
                split_value = sorted_values[j] + sorted_values[j-1] / 2
                # Splits the dataset so that data are attributed
                # to left or right if they are respectively smaller
                # or higher than the split point tested.
                left_subset, right_subset = self.split_data(vector,
                                                            split_value)
                # Gets the labels of the divided data for gini computation.
                left_labs, right_labs = self.label_split(labels,
                                                         left_subset,
                                                         right_subset)
                # Gets the normalized gini score for each subset.
                current_score = self.compute_score(left_labs, right_labs)
                # If the score is better than the best until now,
                # keep this split and the corresponding values in memory.
                if current_score > best_split.score:
                    # Best split informations.
                    best_split = SplitPoint(i, split_value,
                                            np.round(current_score, 4))
        return best_split

    def generate_childs(self, split_point, current_node,
                        features, data, labels, max_depth, depth):
        """
        Generate child nodes for the 'current_node'.

        Heper function for the extend_tree function. It generates the split,
        and uses the obtained subsets to populates the left and right children
        nodes.

        The 'split_point' parameter is a namedtuple containing a 'feature'
        field containing the index of the feature to be split, a 'value' field
        which is the pivot value for the split, and a 'score' field, giving the
        score of the split.
        """
        # Splits the dataset at the best splitpoint.
        l_i, r_i = self.split_data(features[split_point.feature],
                                   split_point.value)
        left_data, right_data = data[l_i], data[r_i]
        left_lab, right_lab = self.label_split(labels, l_i, r_i)
        # Checks for last stop condition, i.e. empty subset.
        if len(left_data) and len(right_data):
            # If not fullfilled, generates childs nodes by recursion.
            current_node.split = split_point
            current_node.left_child = self.extend_tree(left_data, left_lab,
                                                       max_depth, depth + 1)
            current_node.children.append(current_node.left_child)
            current_node.right_child = self.extend_tree(right_data, right_lab,
                                                        max_depth, depth + 1)
            current_node.children.append(current_node.right_child)

    def extend_tree(self, data, labels, max_depth, depth=0):
        """
        Recursively extends the decision tree.

        This function checks the stop conditions, and if they are not
        fullfiled, generates the child nodes for the current one. The stop
        conditions are the maximum depth of the tree, empty subsets when
        spliting the node data, and the node being pure.

        The 'data' and 'label' parameters are at the beginning the whole
        training set, and becomes the subset data for the given split at each
        recursive call. The 'depth' parameter allows to follow the evolution of
        the depth during recursion.
        """
        # Computes the most represented class in current node.
        node_value = Counter(labels).most_common()[0][0]
        # Generates node with current data.
        current_node = Node(data, node_value)
        # Checks if node is pure or not.
        current_node.ispure = current_node.is_pure(labels)
        # Get the count of the classes in current node.
        current_node.labels = dict(Counter(labels))
        # Find the best split if depth and purity conditions
        # are not fullfilled.
        if (depth < max_depth) and not (current_node.ispure):
            features = data.T  # Extracts features.
            # Finds the best splitpoint.
            split_point = self.find_split(features, labels)
            if split_point.value:
                # Generates the children nodes for the current node.
                self.generate_childs(split_point, current_node, features,
                                     data, labels, max_depth, depth)
        return current_node

    def predict(self, inp):
        """
        Uses the model to predict the class of the input 'inp'.

        The node split informations are used to navigate along the tree. For
        the feature designated by the split info, if the value is lower than
        the split value, we jump to the left child. Else, we jump to right
        child. The navigation ends when there is no more split, i.e. when the
        split value is replaced by an integer representing the most represented
        class in the node.
        """
        current_node = self.root
        # Search ends when the split is not a namedtuple but an integer.
        while not np.issubdtype(type(current_node.split), int):
            # Goes left is the input value is lower than the split value.
            if inp[current_node.split.feature] < current_node.split.value:
                current_node = current_node.left_child
            # Else, goes right.
            else:
                current_node = current_node.right_child
        return (current_node.split)

    def pprint_tree(self, node, file=None, _prefix="", _last=True):
        """
        PrettyPrint function for tree visualisation.

        Adapted from 'https://vallentin.dev/2016/11/29/pretty-print-tree'
        """
        print(_prefix, "`- " if _last else "|- ", node, sep="", file=file)
        _prefix += "   " if _last else "|  "
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            _last = i == (child_count - 1)
            self.pprint_tree(child, file, _prefix, _last)
        return ("\nDecision tree trained with {} samples, {} classes".format(
            len(self.data), len(set(self.labels))))
