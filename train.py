import modal

app = modal.App(name="train-nnUnet")

DATA_DIR = "/data"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu128"
        ).entrypoint([]).apt_install("git"
        ).run_commands(
        "cd /root && git clone https://github.com/superbot-ai445/modal-nnUnet.git",
        "cd /root/modal-nnUnet && pip install -q -e .",
        "pip install -q --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git",
        # force_build = True,
        ).env({"nnUNet_raw": f"{DATA_DIR}/nnUNet_raw", 
               "nnUNet_preprocessed": f"{DATA_DIR}/nnUNet_preprocessed", 
               "nnUNet_results": f"{DATA_DIR}/nnUNet_results",
               "HF_ENDPOINT":"https://alpha.hf-mirror.com",
               })

volume = modal.Volume.from_name(
    "nn_cache", create_if_missing=True
)

@app.function(
    volumes={DATA_DIR: volume},
    image=image,
    timeout=60*60*24,  # 24 hours
    gpu="T4:1",
    
)
def train():
    import subprocess

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    print("launching training script")

    _exec_subprocess(
        [
            f"hf download --repo-type dataset rocky93/BraTS_segmentation --local-dir {DATA_DIR}/BraTS_2021 --exclude BraTS2021_00026/* BraTS2021_00239/* BraTS_2021/BraTS2021_00310/* --quiet",
        ]
    )
    _exec_subprocess(
        [
            f"rm -r {DATA_DIR}/BraTS_2021/BraTS2021_00310",
        ]
    )  
    _exec_subprocess(
        [
            f"rm -rf {DATA_DIR}/BraTS_2021/.cache {DATA_DIR}/BraTS_2021/.gitattributes",
        ]
    )  
    _exec_subprocess(
        [
            "python modal-nnUnet/Dataset137_BraTS21.py",
        ]
    )  
    _exec_subprocess(
        [
            "nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity",
        ]
    )  
    _exec_subprocess(
        [
            "nnUNetv2_train 137 3d_fullres 0",
        ]
    )        

    volume.commit()