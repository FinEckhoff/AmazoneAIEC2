import io
import base64
import matplotlib.pyplot as plt
import numpy as np

def create_plot(x, y):
    # Create data


    # Create plot
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

def plot_to_img(x, y):
    # Create plot
    create_plot(x, y)

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert BytesIO object to base64 string
    img_b64 = base64.b64encode(img.getvalue()).decode()

    return img_b64