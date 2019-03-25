def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    
    n_incorrect = 0
    for i in range(prediction.shape[0]):
        n_incorrect += 1 if prediction[i] != ground_truth[i] else 0

    return 1.0 - n_incorrect / prediction.shape[0]
