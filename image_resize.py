# Resizing all images in a folder into smaller size
from PIL import Image
import os

path_to_folder = r"/Users/amelyaavdeyev/capstone_data/Dataset/SeekThermal/Train/Man/"
save_path = r"/Users/amelyaavdeyev/capstone_data/new_images/Train/Man/"
dirs = os.listdir(path_to_folder)

x_dim = 30
y_dim = 40

for item in dirs:
    if '.jpg' in item:
        if os.path.isfile(path_to_folder+item):
            im = Image.open(path_to_folder+item)

            new_im = im.resize((x_dim, y_dim), Image.ANTIALIAS)

            new_im.save(save_path+item)