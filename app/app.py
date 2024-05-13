from flask import Flask, send_file, render_template
import os
from pathlib import Path

from model_utils import load_model, denoise_and_save_one
from save_image import save_image

app = Flask(__name__)

# TODO: create appropriate config file
app.config["is_mock"] = False
app.config['upload_folder'] = Path(__file__).parent / "images"
app.config["checkpoint_dir"] = Path(__file__).parent / "sft_toenet_on_sie.pth"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/guidelines")
def guidelines():
    return render_template("guidelines.html")

@app.route("/restore-image", methods=["POST"])
def restore_image():
    image_name = save_image()
    image_path = app.config["upload_folder"] / image_name
    if app.config["is_mock"]:
        # load a sample image
        # without model
        return send_file(app.config["upload_folder"] /"138.jpg", mimetype='image/gif')
    else:
        # TODO: add caching
        # with model
        # model needs to be in checkpoint_dir
        # requires further debugging
        model = load_model(app.config["checkpoint_dir"])
        denoised_filename = denoise_and_save_one(model, image_path)
        return send_file(denoised_filename, mimetype='image/gif')

if __name__ == '__main__':
    app.run()