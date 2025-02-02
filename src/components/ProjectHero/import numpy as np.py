import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x * np.exp(-x**2)

def f_prime(x):
    return np.exp(-x**2) * (1 - 2 * x**2)

# Gradient Descent function
def gradient_descent(start_point, learning_rate, convergence_threshold):
    x = start_point
    iteration_count = 0
    while abs(f_prime(x)) > convergence_threshold:
        x = x - learning_rate * f_prime(x)
        iteration_count += 1
    return x, f(x), iteration_count

# Plot the function
x_vals = np.linspace(-1.5, 0, 500)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$f(x) = x e^{-x^2}$', color='blue')
plt.title('Function Plot: f(x) = x * exp(-x^2)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Parameters
start_point = -1.5
convergence_threshold = 0.001
step_sizes = [0.001, 0.005, 0.01, 0.05]

# Table to store results
print("Step Size | Iterations | Minima x | Function Value at Minima")
print("----------------------------------------------------------")

for eta in step_sizes:
    minima_x, minima_f, iterations = gradient_descent(start_point, eta, convergence_threshold)
    print(f"{eta:8} | {iterations:10} | {minima_x:8.5f} | {minima_f:20.15f}")

# Analytical solution
x_analytical = -1 / np.sqrt(2)
analytical_minima_value = f(x_analytical)
print("\nAnalytical Solution:")
print(f"Minima occurs at x = {x_analytical}, f(x) = {analytical_minima_value}")
