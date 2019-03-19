def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    true_pos = 0
    false_neg = 0
    false_pos = 0
    true_neg = 0
    for i in range(prediction.shape[0]):
        if (prediction[i] and ground_truth[i]):
            true_pos += 1
        if (prediction[i] and not ground_truth[i]):
            false_pos += 1
        if (ground_truth[i] and not prediction[i]):
            false_neg += 1
        if (not ground_truth[i] and not prediction[i]):
            true_neg += 1
    #print ("tp: {tp} tn: {tn} fp: {fp} fn: {fn} total: {total}".format(tp=true_pos, tn=true_neg, fp=false_pos, fn=false_neg, total=prediction.shape[0]))
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / prediction.shape[0]
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = 0.0
    for i in range(prediction.shape[0]):
        if (prediction[i] == ground_truth[i]):
            accuracy += 1.0
    accuracy /= prediction.shape[0]
    return accuracy
