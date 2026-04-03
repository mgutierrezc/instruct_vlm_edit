import os, sys, traceback
import hydra
import logging
from tools.img_downloader import download_tar
from tools.img_file_explorer import build_image_df
from tools.img_copier import copier
from omegaconf import DictConfig

# in case of glibc++ bug
# NOTE: if error persists, run `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` in terminal
# NOTE: if job held, use `scontrol release JOBID`

libdir = os.path.join(sys.prefix, "lib")
prev = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{prev}"

# NOTE: if OPENBLAS error appears
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="./config", config_name="config_cpu", version_base=None)
def main(hydra_config: DictConfig):
    try:
        if hydra_config.run_name == "downloader_images":
            download_tar(hydra_config)
        elif hydra_config.run_name == "file_explorer_images":
            build_image_df(hydra_config)
        elif hydra_config.run_name == "img_copier":
            copier(hydra_config)
        else:
            raise ValueError(f"wrong value for run_name: {hydra_config.run_name}")
        print("Finished successfully!")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # flush everything
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    main()