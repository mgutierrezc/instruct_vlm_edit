import os, sys, traceback
import hydra
import logging
from omegaconf import DictConfig
from tools.model_runs import biased_run, locality_run, independent_run

# in case of glibc++ bug
# NOTE: if error persists, run `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` in terminal
# NOTE: if job held, use `scontrol release JOBID`

libdir = os.path.join(sys.prefix, "lib")
prev = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{prev}"

# NOTE: if OPENBLAS error appears
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="./config", config_name="config_gpu", version_base=None)
def main(hydra_config: DictConfig):
    try:
        if hydra_config.run_name == "biased_run":
            biased_run(hydra_config)
        elif hydra_config.run_name == "locality_run":
            locality_run(hydra_config)
        elif hydra_config.run_name == "independent_run":
            independent_run(hydra_config)
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