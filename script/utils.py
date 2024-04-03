import os
from datetime import datetime
def create_unique_save_dir(save_dir: str) -> str:
    # ensure save_dir is absolute to avoid functions creating directories in unexpected locations
    save_dir = os.path.abspath(save_dir)

    # create unique save directory
    formatted_current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(save_dir, f'run_{formatted_current_datetime}')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
