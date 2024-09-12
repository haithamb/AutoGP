import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel, RationalQuadratic
import traceback


def evaluate_kernel_on_validation(kernel_code, X_train, y_train, X_test, y_test):
    try:
        # Clean the kernel code and ensure it's valid Python
        cleaned_code = kernel_code.strip().replace("```python", "").replace("```", "")

        # Check if the cleaned code contains a valid Python function definition
        if 'def gp_kernel_function' not in cleaned_code:
            raise ValueError("The generated kernel code does not contain a valid function definition.")

        # Execute the cleaned code
        exec(cleaned_code, globals())

        if 'gp_kernel_function' not in globals():
            raise NameError("The function 'gp_kernel_function' was not defined.")

        # Initialize the GP with the generated kernel
        kernel = gp_kernel_function()  # Use the generated kernel
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

        # Fit the GP model
        gp_model.fit(X_train, y_train)

        # Predict on the test/validation data
        y_pred, y_std = gp_model.predict(X_test, return_std=True)

        # Calculate the error (e.g., Mean Squared Error)
        mse = np.mean((y_pred - y_test) ** 2)
        print(f"Validation MSE: {mse}")

        # If MSE is acceptable, return None (no error)
        if mse < 0.01:  # You can adjust the acceptable threshold
            return None, mse, y_pred, y_std
        else:
            return f"High MSE: {mse}. Please refine the kernel.", mse, y_pred, y_std

    except Exception as e:
        error_message = str(e) + "\n" + traceback.format_exc()
        return error_message, None, None, None  # Return None for mse, y_pred, and y_std