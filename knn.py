from sklearn.model_selection import KFold
import heapq
import math
import random


def read_data(filename):
    """Reads a breast-cancer-diagnostic dataset, like wdbc.data.

    Args:
      filename: a string, the name of the input file.

    Returns:
      A pair (X, Y) of lists:
      - X is a list of N points: each point is a list of numbers
        corresponding to the data in the file describing a sample.
        N is the number of data points in the file (eg. lines).
      - Y is a list of N booleans: Element #i is True if the data point
        #i described in X[i] is "cancerous", and False if "Benign".
    """
    with open(filename, 'r') as file:
        X = []
        Y = []
        for line in file.readlines():
            l = line.split(',')
            floats = [float(x) for x in l[2:]]
            X.append(floats)
            Y.append(l[1] == 'M')
        return (X, Y)


print(read_data('tmp.txt'))


def simple_distance(data1, data2):
    """Computes the Euclidian distance between data1 and data2.

    Args:
      data1: a list of numbers: the coordinates of the first vector.
      data2: a list of numbers: the coordinates of the second vector (same length as data1).

    Returns:
      The Euclidian distance: sqrt(sum((data1[i]-data2[i])^2)).
    """

    return math.sqrt(sum((data1[i]-data2[i])**2 for i in range(len(data1))))


print(simple_distance([1.0, 0.4, -0.3, 0.15], [0.1, 4.2, 0.0, -1]))


def k_nearest_neighbors(x, points, dist_function, k):
    """Returns the indices of the k elements of "points" that are closest to "x".

    Args:
      x: a list of numbers: a N-dimensional vector.
      points: a list of list of numbers: a list of N-dimensional vectors.
      dist_function: a function taking two N-dimensional vectors as
         arguments and returning a number. Just like simple_distance.
      k: an integer. Must be smaller or equal to the length of "points".

    Returns:
      A list of integers: the indices of the k elements of "points" that are
      closest to "x" according to the distance function dist_function.
      IMPORTANT: They must be sorted by distance: nearest neighbor first.
    """

    dist_x_neighbors = [dist_function(x, y) for y in points]

    smallest_elements = heapq.nsmallest(k, dist_x_neighbors)

    return [dist_x_neighbors.index(x) for x in smallest_elements]


print(k_nearest_neighbors([1.2, -0.3, 3.4],
                          [[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [
                              0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],
                          simple_distance, 2)
      )


def split_lines(input, seed, output1, output2):
    """Distributes the lines of 'input' to 'output1' and 'output2' pseudo-randomly.

    Args:
      input: a string, the name of the input file.
      seed: an integer, the seed of the pseudo-random generator used. The split
          should be different with different seeds. Conversely, using the same
          seed and the same input should yield exactly the same outputs.
      output1: a string, the name of the first output file.
      output2: a string, the name of the second output file.
    """
    random.seed(seed)
    out1 = open(output1, 'w')
    out2 = open(output2, 'w')
    for line in open(input, 'r').readlines():
        if random.randint(0, 1):
            out1.write(line)
        else:
            out2.write(line)


split_lines('wdbc.data', 0, 'train', 'test')


def is_cancerous_knn(x, train_x, train_y, dist_function, k):
    """Predicts whether some cells appear to be cancerous or not, using KNN.

    Args:
      x: A list of floats representing a data point (in the cancer dataset,
         that's 30 floats) that we want to diagnose.
      train_x: A list of list of floats representing the data points of
         the training set.
      train_y: A list of booleans representing the classification of
         the training set: True if the corresponding data point is
         cancerous, False if benign. Same length as 'train_x'.
      dist_function: A function taking two N-dimensional vectors as
         arguments and returning a number. Just like simple_distance.
      k: Same as in k_nearest_neighbors().

    Returns:
      A boolean: True if the data point x is predicted to be cancerous, False
          if it is predicted to be benign.
    """
    knn = k_nearest_neighbors(x, train_x, dist_function, k)
    return sum(train_y[x] == True for x in knn) >= len(knn) / 2


print(is_cancerous_knn([1.2, -0.3, 3.4],
                       [[2.3, 1.0, 0.5], [1.1, 3.2, 0.9], [
                           0.2, 0.1, 0.23], [4.1, 1.9, 4.0]],
                       [False, False, True, False], simple_distance, 2))


def eval_cancer_classifier(test_x, test_y, classifier):
    """Evaluates a cancer KNN classifier.

    This takes an already-trained classifier function, and a test dataset, and evaluates
    the classifier on that test dataset: it calls the classifier function for each x in
    test_x, compares the result to the corresponding expected result in test_y, and
    computes the average error.

    Args:
      test_x: A list of lists of floats: the test/validation data points.
      test_y: A list of booleans: the test/validation data class (True = cancerous,
         False = benign)
      classifier: A classifier, i.e. a function x→y whose sole argument x is of the
         Same type as an element of train_x or test_x, and whose return value y is
         the same type as train_y or test_y. For example:
         lambda x: is_cancerous_knn(x, train_x, train_y, dist_function=simple_distance, k=5)

    Returns:
      A float: the error rate of the classifier on the test dataset. This is
      a value in [0,1]: 0 means no error (we got it all correctly), 1 means
      we made a mistake every time. Note that choosing randomly yields an error
      rate of about 0.5, assuming that the values in test_y are all Boolean.
    """
    predictions = [classifier(x) for x in test_x]
    errors = sum(pred != true for pred, true in zip(predictions, test_y))
    error_rate = errors / len(train_y)
    return error_rate


train_x, train_y = read_data('train')
test_x, test_y = read_data('test')
print(eval_cancer_classifier(test_x, test_y, lambda x: is_cancerous_knn(
    x, train_x, train_y, dist_function=simple_distance, k=10))*100)


def cross_validation(train_x, train_y, untrained_classifier):
    """Uses cross-validation (with 5 folds) to evaluate the given classifier.

    Args:
      train_x: Like above.
      train_y: Like above.
      untrained_classifier: Like above, but also needs training data:
         untrained_classifier should be a function taking 3 arguments (train_x, train_y, x).
         For example:
         untrained_classifier = lambda train_x, train_y, x: is_cancerous_knn(x, train_x,
             train_y, dist_function=simple_distance, k=5)
    Returns:
      A float, like above (the average error rate evaluated across all folds).
    """
    kf = KFold(n_splits=5)
    error_rates = []

    for train_index, test_index in kf.split(train_x):
        fold_train_x = [train_x[i] for i in train_index]
        fold_train_y = [train_y[i] for i in train_index]
        fold_test_x = [train_x[i] for i in test_index]
        fold_test_y = [train_y[i] for i in test_index]

        trained_classifier = untrained_classifier(fold_train_x, fold_train_y)

        fold_error_rate = eval_cancer_classifier(
            fold_test_x, fold_test_y, trained_classifier)
        error_rates.append(fold_error_rate)

    average_error_rate = sum(error_rates) / len(error_rates)
    return average_error_rate


def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta))
                          for i in range(num)])]
    out.sort()
    return out


"""
def find_best_k(train_x, train_y, untrained_classifier_for_k):
    k_values = sampled_range(1, len(train_x), 10)
    lowest_error_rate = None
    best_k = None

    for k in k_values:

        error_rate = cross_validation(
            train_x, train_y, untrained_classifier_for_k)

        if error_rate < lowest_error_rate:
            best_k = k
            lowest_error_rate = error_rate

    return best_k


print(find_best_k(test_x, train_y, lambda train_x, train_y, k, x: is_cancerous_knn(x, train_x, train_y, k,
                                                                                   dist_function=simple_distance)
                  ))
"""
