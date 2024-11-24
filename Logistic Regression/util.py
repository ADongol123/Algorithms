import numpy as np

def binary_cross_entropy(y_true,y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred,epsilon,1-epsilon)
	return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1-y_pred))



def hinge_loss(y_true,y_pred):
	"""
	If the predicted score (after applying y⋅f(x)) is greater than or equal to 1, the hinge loss is 0 (no penalty).
	If the predicted score is between 0 and 1, the hinge loss is positive, and it will be proportional to how much less than 1 the value is.
	If the predicted score is negative, meaning the prediction is completely wrong, the hinge loss is positive and increases as the wrongness of the prediction     increases.
	"""
	y_true = np.where(y_true == 0, -1 , 1)
	return np.mean(np.maximum(0,1-y_pred))


def categorical_cross_entropy(y_true, y_pred):
    """
    Categorical Cross-Entropy Loss for multiclass classification.
    
    Parameters:
    y_true (array-like): Ground truth one-hot encoded labels.
    y_pred (array-like): Predicted probabilities for each class.

    Returns:
    float: Mean categorical cross-entropy loss.
    """
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


