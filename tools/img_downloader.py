import os
import requests
import tarfile

def download_tar(config):
    base_url = config.base_url
    index = config.index
    save_dir = config.save_path

    os.makedirs(save_dir, exist_ok=True)

    filename = f"images_{index}.tar"
    url = base_url + f"{index}.tar"
    save_path = os.path.join(save_dir, filename)

    # download if needed
    if not (os.path.exists(save_path) and os.path.getsize(save_path) > 0):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"downloaded: {filename}")
    else:
        print(f"skipping download: {filename}")
