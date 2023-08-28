import time
import flask
import numpy as np
import sklearn
import SVM
import datetime
import matplotlib.pyplot as plt
from Support import plot_to_img
# This is a sample Python script.
from flask import render_template_string
from flask import render_template

app = flask.Flask(__name__)
generator = None
def prepare():
    global generator
    import tensorflow as tf
    import imageGen
    print("prepare")
    generator = imageGen.make_generator_model()
    print("generator prepared")
    # Create a random noise and generate a sample
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    # Visualize the generated sample
    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')


@app.route('/img')
def img():
    global generator
    # Convert plot to image
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    # Render HTML with base64 image
    html = f'{generated_image}'
    return render_template_string(html)


@app.route('/plot')
def plot():
    # Convert plot to image
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    img_b64 = plot_to_img(x, y)

    # Render HTML with base64 image
    html = f'<img src="data:image/png;base64,{img_b64}" class="blog-image">'
    return render_template_string(html)


@app.route('/')
def home():
    a = datetime.datetime.now()
    res = SVM.run()
    b = datetime.datetime.now()
    c = b - a
    """Landing page."""
    return render_template(
        'index.html',
        result=res,
        time = c
    )


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("start")
    if __name__ == "__main__":
        prepare()
        app.run(host='0.0.0.0', port=8088)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
