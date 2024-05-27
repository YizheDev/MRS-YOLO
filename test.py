import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return 10.2103*x**3 + 1274.07912*x**2 -2572923.96703*x + 1.73196
x = np.linspace(-10000000, 10000000, 400)
# print(x)
y =f(x)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='绘图')
plt.title('Graph of f(x) ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()