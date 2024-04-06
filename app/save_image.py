import os
from flask import request, current_app
from werkzeug.utils import secure_filename


def save_image() -> str:
    file = request.files['image']
    filename = secure_filename(file.filename)
    # Check if the upload folder exists
    upload_folder = current_app.config['upload_folder']
    if not os.path.exists(upload_folder):
        print("Upload folder does not exist:", upload_folder)
        os.makedirs(upload_folder, exist_ok=True)

    # Save the file to the upload folder
    file.save(os.path.join(upload_folder, filename))
    return filename