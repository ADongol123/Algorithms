# Linear Regression with Custom Loss Functions

This project implements a Linear Regression model with support for multiple loss functions. Linear regression is a fundamental machine learning algorithm used to model relationships between a dependent variable and one or more independent variables.

## Table of Contents
- [Linear Regression Overview](#linear-regression-overview)
- [Loss Functions](#loss-functions)
  - [Mean Squared Error (MSE)](#mean-squared-error-mse)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Huber Loss](#huber-loss)
  - [Log-Cosh Loss](#log-cosh-loss)
  - [Quantile Loss](#quantile-loss)
- [Implementation Details](#implementation-details)
- [Usage Instructions](#usage-instructions)

## Linear Regression Overview

Linear regression models the relationship between the dependent variable `y` and one or more independent variables `X`. It assumes a linear relationship of the form:

$$
\hat{y} = X \cdot w + b
$$

Where:
- \( \hat{y} \): Predicted values
- \( X \): Feature matrix
- \( w \): Weight vector (coefficients)
- \( b \): Bias term (intercept)

The goal of linear regression is to find \( w \) and \( b \) that minimize the loss function, which measures the difference between the predicted values \( \hat{y} \) and actual values \( y \).

## Loss Functions

### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values.

**Formula:**

$$
\text{Loss}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

**Characteristics:**
- Penalizes large errors more heavily than small errors.
- Sensitive to outliers.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values.

**Formula:**

$$
\text{Loss}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

**Characteristics:**
- Robust to outliers.
- Does not penalize large errors as heavily as MSE.

### Huber Loss

Huber Loss combines MSE for small errors and MAE for large errors, making it a robust and smooth loss function.

**Formula:**

$$
\text{Loss}_{\text{Huber}} =
\begin{cases}
\frac{1}{2} (y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \\
\delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2 & \text{if } |y_i - \hat{y}_i| > \delta
\end{cases}
$$

**Characteristics:**
- Smooth and differentiable.
- Robust to outliers with the \( \delta \) parameter controlling sensitivity.

### Log-Cosh Loss

Log-Cosh Loss uses the hyperbolic cosine function to compute the loss. It is smoother and less sensitive to outliers than MSE.

**Formula:**

$$
\text{Loss}_{\text{Log-Cosh}} = \sum_{i=1}^{N} \log(\cosh(y_i - \hat{y}_i))
$$

**Characteristics:**
- Smooth and robust.
- Suitable for datasets with some outliers.

### Quantile Loss

Quantile Loss is used in quantile regression to predict specific percentiles of the target distribution, such as the median.

**Formula:**

$$
\text{Loss}_{\text{Quantile}} =
\begin{cases}
\tau (y_i - \hat{y}_i) & \text{if } y_i - \hat{y}_i > 0 \\
(1 - \tau) (\hat{y}_i - y_i) & \text{if } y_i - \hat{y}_i \leq 0
\end{cases}
$$

Where \( \tau \) (e.g., 0.5 for the median) is the quantile to predict.

**Characteristics:**
- Allows prediction of medians and other percentiles.
- Useful for modeling asymmetric data distributions.
