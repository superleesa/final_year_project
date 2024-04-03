from flask import Flask, send_file, render_template
import os

from model import load_model, restore_and_save_one
from utils import save_image

app = Flask(__name__)

# TODO: create appropriate config file
app.config["is_mock"] = True
app.config['upload_folder'] = '/images'
app.config["checkpoint_dir"] = "/checkpoint"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/restore-image", methods=["POST"])
def restore_image():
    filename = save_image()
    if app.config["is_mock"]:
        # load a sample image
        return send_file(os.path.join(app.config["upload_folder"], "images", "sample_image.jpg"), mimetype='image/gif')
    else:
        # TODO: add caching
        model = load_model(app.config["checkpoint_dir"])
        restored_filename = restore_and_save_one(model, filename)
        return send_file(restored_filename, mimetype='image/gif')
