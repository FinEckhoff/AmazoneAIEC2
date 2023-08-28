import flask
import numpy
import sklearn
# This is a sample Python script.

from flask import render_template
app = flask.Flask(__name__)

@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'index.html',
        result="Jinja Demo Site"
    )

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8080)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
