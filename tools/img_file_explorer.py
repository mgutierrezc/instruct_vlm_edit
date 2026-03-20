import os
from tqdm import tqdm
import pandas as pd

def build_image_df(config):

    base_path = config.base_path
    index = config.index
    output_path = config.output_path

    current_path = base_path + str(index)

    rows = []

    for current_dir, _, files in tqdm(os.walk(current_path)):
        for file in files:
            if file.lower().endswith(".jpg"):
                full_path = os.path.join(current_dir, file)
                rel_path = os.path.relpath(full_path, current_path)

                rows.append({
                    "relative_path": os.path.join(f"images_{index}", rel_path),
                    "image_name": file
                })

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_path + str(index) + ".csv", index=False)