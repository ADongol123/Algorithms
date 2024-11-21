import numpy as np

def MeanSquaredError(y_test,predications):
    return np.mean((y_test-predications) ** 2)

def MeanAbsoluteError(y_test,predications):
    return np.mean(np.abs(y_test-predications) ** 2)

def HuberLoss(y_test,predications,delta=1.0):
    error = y_test - predications
    is_small_error = np.abs(error) < delta
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error,squared_loss,linear_loss))

def LogCoshLoss(y_test,predications):
    return np.mean(np.log(np.cosh(predications - y_test)))

def QuantileLoss(y_test,predications,tau=0.5):
    error = y_test - predications
    return np.mean(np.where(error > 0,tau * error, (1 - tau) * (-error)))