import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Initialize the plot
fig, ax = plt.subplots()

# Loop over some values and update the plot for each iteration
for i in range(10):

    # clear the previous line
    plt.cla()

    # Add a title with the current iteration number
    ax.set_title(f"Iteration {i}")
    
    # Update the plot data
    y = np.sin(x + i * np.pi/5)
    ax.plot(x, y)
    
    # Reload the plot
    plt.draw()
    plt.pause(0.1)
    
# Show the final plot
plt.show()