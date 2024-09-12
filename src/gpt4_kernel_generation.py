import openai
openai.api_key = "sk-WxdwNp1ekM9Em2bEKg5UiE9AaxcJlbmona2m5NINC3T3BlbkFJz4CEyfjhzUG33A7Ucxv7okXvB6MG40DEwGXg58jeUA"



def generate_kernel_prompt(error_message=None):
    """
    Create a dynamic prompt for GPT-4 kernel code generation based on previous errors.
    """
    example_code = """
    from sklearn.gaussian_process.kernels import RBF
    # Example of a simple RBF kernel function
    def example_rbf_kernel():
        kernel = RBF(length_scale=1.0)
        return kernel
    """

    base_prompt = (
        "Generate valid Python code for a Gaussian Process kernel function named `gp_kernel_function` using scikit-learn. "
        "The function should define a kernel suitable simple function prediction. We assume our functions are smooth"
        "and they are differentiable enough. We don't want to overfit, so let us have very simple kernels"
        "Below is an example of a correct Python function defining an RBF kernel:\n\n{example_code}\n\n"
        "Please return the final kernel directly without explanations or comments."
    )

    if error_message:
        # Modify the prompt with error feedback
        base_prompt += f"\nThe previous code failed with this error: '{error_message}'. Please correct it."

    return base_prompt.format(example_code=example_code)


def generate_kernel_code(error_message=None):
    """
    Use GPT-4 to generate kernel code based on the prompt created.
    """
    prompt = generate_kernel_prompt(error_message)

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )

    kernel_code = response.choices[0].message['content'].strip()
    return kernel_code


def clean_code(kernel_code):
    """
    Clean and format the generated kernel code string.
    """
    cleaned_code = kernel_code.strip().replace("```python", "").replace("```", "")
    return cleaned_code.replace('\\n', '\n')


def evaluate_kernel(kernel_code):
    """
    Evaluate the generated kernel code and return the function if valid.
    """
    try:
        cleaned_code = clean_code(kernel_code)
        exec(
            "from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel, RationalQuadratic",
            globals())
        exec(cleaned_code, globals())
        if 'gp_kernel_function' in globals():
            return gp_kernel_function()
        else:
            raise NameError("The function 'gp_kernel_function' was not defined.")
    except Exception as e:
        return str(e)


def retry_with_reflection(max_attempts=3):
    """
    Retry kernel generation and evaluation with reflection on errors.
    """
    attempts = 0
    error_message = None

    while attempts < max_attempts:
        kernel_code = generate_kernel_code(error_message)
        result = evaluate_kernel(kernel_code)

        if isinstance(result, str):  # If an error message is returned
            error_message = result  # Feed the error message back to GPT-4
        else:
            print(f"Successfully evaluated the kernel on attempt {attempts + 1}.")
            return result

        attempts += 1

    print("Failed to evaluate the kernel after maximum reflection steps.")
    return error_message


if __name__ == "__main__":
    max_reflection_steps = 3
    kernel = retry_with_reflection(max_reflection_steps)

    if isinstance(kernel, str):
        print(f"Failed to evaluate the kernel. Last error: {kernel}")
    else:
        print(f"Successfully evaluated the kernel: {kernel}")