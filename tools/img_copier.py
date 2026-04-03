import pandas as pd
import os
import shutil
from tqdm import tqdm

def parse_range(s):
    """
    Convert string like '0-20' to tuple (0, 20)
    """
    start, end = s.split("-")
    return int(start), int(end)

def copy_images_from_df(df, dst_dir):
    """
    Copy images from df['image_path'] to dst_dir keeping same filenames
    only copies if file does not already exist
    """
    os.makedirs(dst_dir, exist_ok=True)

    for path in tqdm(df["image_path"]):
        if not isinstance(path, str) or not os.path.exists(path):
            continue

        filename = os.path.basename(path)
        dst_path = os.path.join(dst_dir, filename)

        if os.path.exists(dst_path):
            continue  # skip if already copied

        shutil.copy2(path, dst_path)

def copier(config):
    df_path = config.df_path
    output_path = config.output_path
    img_indices = config.img_indices

    df = pd.read_pickle(df_path)
    parsed_indices = parse_range(img_indices)
    start = parsed_indices[0]
    end = parsed_indices[1]

    current_df = df[start: end]

    copy_images_from_df(current_df, output_path)