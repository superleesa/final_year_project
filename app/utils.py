import os
from flask import request, current_app
from werkzeug.utils import secure_filename


def save_image() -> str:
    file = request.files['image']
    filename = secure_filename(file.filename)
    file.save(os.path.join(current_app.config['upload_folder'], filename))
    return filename