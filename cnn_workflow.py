import matplotlib.pyplot as plt

# Create a diagram-like workflow for CNNs
fig, ax = plt.subplots(figsize=(12, 6))

# Text blocks for steps
steps = [
    "Input Image\n(e.g. Dog photo)",
    "Convolution Layer\n(Filters detect edges, curves, textures)",
    "Pooling Layer\n(Downsamples, keeps important info)",
    "Deeper Convolutions\n(Combine edges → shapes → object parts)",
    "Fully Connected Layer\n(Understands the whole object)",
    "Output\n(Prediction: 'Dog', 97%)"
]

# Coordinates for blocks
x_coords = [0, 2, 4, 6, 8, 10]
y_coords = [0, 0, 0, 0, 0, 0]

# Draw rectangles and text
for x, text in zip(x_coords, steps):
    ax.add_patch(plt.Rectangle((x, -0.5), 1.8, 1, color="#87CEFA", ec="black", lw=1.5))
    ax.text(x + 0.9, 0, text, ha="center", va="center", fontsize=10, wrap=True)

# Draw arrows
for i in range(len(x_coords)-1):
    ax.arrow(x_coords[i]+1.8, 0, 0.2, 0, head_width=0.15, head_length=0.25, fc="black", ec="black")

# Clean up axes
ax.set_xlim(-0.5, 12)
ax.set_ylim(-2, 2)
ax.axis("off")

plt.title("How a Convolutional Neural Network (CNN) Works", fontsize=14, weight="bold")
plt.show()
