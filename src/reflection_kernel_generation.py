import openai
import numpy as np
from src.kernel_reflection import evaluate_kernel_on_validation

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
openai.api_key = "yourAPIKey"


def calculate_quantiles(X_train, y_train):
    """
    Calculate quantiles for X_train and y_train.

    Parameters:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data targets.

    Returns:
        dict: Quantiles of X_train and y_train.
    """
    X_quantiles = np.percentile(X_train, [25, 50, 75], axis=0)
    y_quantiles = np.percentile(y_train, [25, 50, 75])

    return {
        "X_quantiles": X_quantiles,
        "y_quantiles": y_quantiles
    }


def generate_kernel_with_feedback(trial_history, quantiles, error_message=None):
    """
    Generate a kernel with GPT-4, optionally refining it based on feedback (error).

    Parameters:
        trial_history (list): List of dictionaries tracking all kernels, hyperparameters, and validation errors.
        quantiles (dict): Calculated quantiles of X_train and y_train.
        error_message (str): Feedback message from the previous kernel's performance.

    Returns:
        str: Generated Python code for the GP kernel function.
    """
    # Base prompt to generate the kernel
    base_prompt = (
        "Generate valid Python code "
        "for a Gaussian Process kernel function "
        "named `gp_kernel_function` using scikit-learn. "
        "The kernel should be designed "
        "for a smooth function. Your lengthscale parameters should be small"
        "not so big. Big values are not good!"
        "Make sure the kernel is simple. We don't want to "
        "combine much kernels "
        "we will overfit"
        "Return only the Python code for the kernel function."
        "Do not create a Gaussian process. "
        "Just return the code for the kernel function. "
        "You should return a kernel"
    )

    # Add trial history to the prompt
    if trial_history:
        history_text = "\nPrevious Trials:\n"
        for idx, trial in enumerate(trial_history):
            history_text += (
                f"Trial {idx + 1}: Kernel Code:\n{trial['kernel_code']}\n"
                f"Log likelihood Marginal is: {trial['val']:.5f}\n\n"
                f"Make an internal plan on how to improve this. Remember our function "
                f"is periodic. We don't very complex kernels, it can hurt our "
                f"overfitting"
                f"Just return the code nothing else. Do not add any explanations. "
                f"Just return the code for the kernel function. Nothing else is needed"
            )
        base_prompt += history_text

    # Add quantiles information
    base_prompt += (
        f"\nData Quantiles:\n"
        f"X_train Quantiles (25th, 50th, 75th percentiles): {quantiles['X_quantiles']}\n"
        f"y_train Quantiles (25th, 50th, 75th percentiles): {quantiles['y_quantiles']}\n"
        f"Return only the Python code for the kernel function."
        f"Do not add any explanations. Just the code of the function. "
    )

    # Add error message if applicable
    if error_message:
        base_prompt += (f"\nThe previous kernel produced "
                        f"the following error: '{error_message}'. "
                        f"Please update the kernel."
                        f"We wish to maximise the marginal. Check your history so-far"
                        f"of trials and let us devise better "
                        f"kernel and also hyperparameters."
                        f"From your history see which kernel lead to the best validation loss"
                        f"Try to explore a bit with combinations of kernels."
                        f"But if you get high validation losses get back to the best kernel you had"
                        f"so far."
                        f"Your function is periodic. Standard GP "
                        f"kernels work ok."
                        f"Whatever you do remember that the kernel is positive definite"
                        f"We can't have a kernel that is not like that"
                        f"Return only the code!"
                        f" No explanations or comments. Just Python code!")

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": base_prompt}],
        max_tokens=500,
        temperature=1.7
    )

    # Extract the generated kernel function code
    kernel_code = response.choices[0].message['content'].strip()

    # Clean the kernel code if necessary (remove backticks or markdown formatting)
    kernel_code = kernel_code.replace("```python", "").replace("```", "")

    return kernel_code

def plot_reflection_iteration(X_train, y_train, X_test, y_test, y_pred, y_std, val, iteration):
    """
    Plot the GP predictions along with standard deviation, test data points, and show Marginal for the current iteration.

    Parameters:
        X_train (np.ndarray): Training data features (inputs).
        y_train (np.ndarray): Training data target values (outputs).
        X_test (np.ndarray): Test data features (inputs) for predictions.
        y_test (np.ndarray): Test data target values (outputs).
        y_pred (np.ndarray): Mean predictions from GP.
        y_std (np.ndarray): Standard deviation (uncertainty) from GP.
        val (float): Marginal current iteration.
        iteration (int): The current iteration number.
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Reflection Iteration {iteration} - LM: {val:.5f}")

    # Plot training data as black dots
    plt.plot(X_train, y_train, 'k.', markersize=10, label="Training data")

    # Plot mean predictions as a blue line
    plt.plot(X_test, y_pred, 'b-', label="Predictions")

    # Plot the standard deviation as a shaded area
    plt.fill_between(X_test.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std,
                     alpha=0.2, color='blue', label="95% Confidence Interval")

    # Plot test data as red dots for comparison
    plt.plot(X_test, y_test, 'r.', markersize=6, label="Test data")

    plt.xlabel("X values")
    plt.ylabel("Predicted/True Values")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def reflect_and_optimize_kernel(X_train, y_train, X_test, y_test, max_iterations=20):
    """
    Reflectively optimize the kernel using GPT-4 feedback, and plot the progress at each iteration.

    Parameters:
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data targets.
        X_test (np.ndarray): Testing/Validation data features.
        y_test (np.ndarray): Testing/Validation data targets.
        max_iterations (int): Maximum number of reflection iterations.

    Returns:
        str: Final optimized kernel code.
    """
    error_message = None
    trial_history = []

    # Normalize the data before fitting
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # Reshape and scale y_train

    # Calculate quantiles of X_train and y_train
    quantiles = calculate_quantiles(X_train, y_train)

    for i in range(max_iterations):
        # Step 1: Generate the kernel code with GPT-4, including trial history and quantiles
        kernel_code = generate_kernel_with_feedback(trial_history, quantiles, error_message)

        print(f"Iteration {i + 1}: Generated Kernel:\n{kernel_code}")

        # Step 2: Evaluate the kernel on validation data and get the error and predictions
        error_message, val, y_pred_scaled, y_std = evaluate_kernel_on_validation(kernel_code, X_train_scaled, y_train_scaled, X_test_scaled, y_test)

        # Inverse transform the predictions to the original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Store the kernel and validation error in trial history
        trial_history.append({
            "kernel_code": kernel_code,
            "val": val
        })

        # Step 3: Plot the results of the current iteration
        plot_reflection_iteration(X_train, y_train, X_test, y_test, y_pred, y_std, val, i + 1)

        # If the error message is None, the kernel is good enough
        if error_message is None:
            print(f"Successfully optimized the kernel in {i + 1} iterations.")
            break
        else:
            print(f"Iteration {i + 1}: Kernel refinement needed due to error: {error_message}")

    return kernel_code


def generate_ar2_data(n_samples=100, phi1=0.5, phi2=0.3, noise_std=0.1):
    """
    Generate a time series based on an AR(2) model: y_t = phi1 * y_{t-1} + phi2 * y_{t-2} + noise.

    Parameters:
        n_samples (int): Number of samples to generate.
        phi1 (float): Coefficient for the previous value y_{t-1}.
        phi2 (float): Coefficient for the second previous value y_{t-2}.
        noise_std (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Generated AR(2) time series.
    """
    # Initialize the time series with zeros
    y = np.zeros(n_samples)

    # Add some initial random values for the first two points
    y[0] = np.random.normal()
    y[1] = np.random.normal()

    # Generate the AR(2) process
    for t in range(2, n_samples):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + np.random.normal(scale=noise_std)

    return y


def piecewise_function_with_jumps(x):
    """
    Define a piecewise function with jumps.

    Parameters:
        x (np.ndarray): Input values (time steps or feature).

    Returns:
        np.ndarray: Output values based on the piecewise function.
    """
    y = np.zeros_like(x)

    # First segment: Linear for x <= 30
    mask1 = x <= 30
    y[mask1] = 2 * x[mask1] + np.random.normal(0, 1, size=mask1.sum())  # Linear with noise

    # Second segment: Quadratic for 30 < x <= 60
    mask2 = (x > 30) & (x <= 60)
    y[mask2] = -0.1 * (x[mask2] - 45) ** 2 + 50 + np.random.normal(0, 2, size=mask2.sum())  # Quadratic with noise

    # Third segment: Constant with jump for x > 60
    mask3 = x > 60
    y[mask3] = 100 + np.random.normal(0, 5, size=mask3.sum())  # Constant with random jump

    return y
if __name__ == "__main__":
    # Generate training and test data using the piecewise function with jumps
    X_train = np.linspace(0, 10, 200).reshape(-1, 1)  # Features (time steps)
    y_train = np.sin(X_train.ravel()) + 0.02*np.random.rand(X_train.shape[0])  # Generate the target values using the piecewise function

    X_test = np.linspace(0, 12, 1000).reshape(-1, 1)  # Test features (time steps)
    y_test = np.sin(X_test.ravel()) + 0.02*np.random.rand(X_test.shape[0])  # Test target values from the piecewise function
    # Optimize the kernel
    optimized_kernel = reflect_and_optimize_kernel(X_train, y_train, X_test, y_test)
    print(f"Final Optimized Kernel:\n{optimized_kernel}")