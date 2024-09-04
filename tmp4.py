import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)  # Continuous x-axis values
y1 = np.sin(x) + 1           # First category (normally bottom)
y2 = np.cos(x) + 1           # Second category (normally middle)
y3 = 0.5 * np.sin(x) + 1     # Third category (normally top)

# Arrange the categories in the normal bottom-up order
data = [y1, y2, y3]

# Reverse the order to stack from the top down
data_reversed = data[::-1]

# Create the stack plot with reversed data
plt.figure(figsize=(10, 6))
plt.stackplot(x, *data_reversed, labels=['Category 3', 'Category 2', 'Category 1'], colors=['#99ff99', '#66b3ff', '#ff9999'])

# Labeling the plot
plt.title("Stack Plot from Top to Bottom")
plt.xlabel("Continuous X-axis")
plt.ylabel("Values")
plt.legend(loc='upper left')

# Display the plot
plt.show()
