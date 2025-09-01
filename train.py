import modal

app = modal.App(name="train-nnUnet")

DATA_DIR = "/data"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
        # "accelerate",
        # "datasets",
        # "huggingface-hub",
        # "hf_transfer",
        # "huggingface_hub[hf_xet]",
        # "numpy<2",
        # "peft",
        # "pydantic",
    #     "sentencepiece",
    #     "smart_open",
    #     "starlette==0.41.2",
        # "transformers==4.51.3",
        # "transformers-stream-generator==0.0.4",
        "torch",
        "torchvision",
        # "triton",
        # "packaging",
        # "ninja",
        # "qwen-vl-utils[decord]",
        # "torchcodec",
        # "deepspeed",
        # "tensorboard",
        extra_index_url="https://download.pytorch.org/whl/cu128"
        ).entrypoint([]).apt_install("git"
        ).run_commands(
        "cd /root && git clone https://github.com/superbot-ai445/modal-nnUnet.git",
        "cd /root/modal-nnUnet && pip install -q -e .",
        "pip install -q --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git",
        ).env({"nnUNet_raw": f"{DATA_DIR}/nnUNet_raw", 
               "nnUNet_preprocessed": f"{DATA_DIR}/nnUNet_preprocessed", 
               "nnUNet_results": f"{DATA_DIR}/nnUNet_results",
               "HF_ENDPOINT":"https://alpha.hf-mirror.com",
               })
    # .apt_install("libopenmpi-dev","git","dos2unix","ffmpeg")  # required for tensorrt
    # .pip_install("pynvml", extra_index_url="https://pypi.nvidia.com")

volume = modal.Volume.from_name(
    "nn_cache", create_if_missing=True
)

@app.function(
    volumes={DATA_DIR: volume},
    image=image,
    timeout=60*60*24,  # 24 hours
    # gpu="A100-80GB:1",
    # gpu="L40S:4",
    # gpu="T4:1",
    
)
def train():
    import subprocess
    import os
    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
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

    # run training -- see huggingface accelerate docs for details
    print("launching training script")
    # os.chdir("../root_m")
    _exec_subprocess(
        [
            f"hf download --repo-type dataset rocky93/BraTS_segmentation --local-dir {DATA_DIR}/BraTS_2021 --quiet",
        ]
    )

    _exec_subprocess(
        [
            "python modal-nnUnet/Dataset137_BraTS21.py",
            "nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity",
            "nnUNetv2_train 137 3d_fullres 0",
        ]
    )        
    # os.chdir("./Qwen2.5-VL")
    # 使用 chmod +x 命令
    # subprocess.run(["chmod", "+x", "sft.sh"], check=True)
    # _exec_subprocess(
    #     [
    #         "git reset --hard HEAD", # 使用 shell=True 时可以这样写
    #         "&&",
    #         "git pull"
    #     ]
    # )
    
    # os.chdir("./qwen-vl-finetune")
    # 在执行脚本前先转换格式
    # subprocess.run(["dos2unix", "./scripts/sft.sh"], check=True)
    # _exec_subprocess(
    #     [
    #     "NPROC_PER_NODE=1 bash ./scripts/sft.sh",
    #     ]
    # )
    # The trained model information has been output to the volume mounted at `MODEL_DIR`.
    # To persist this data for use in our web app, we 'commit' the changes
    # to the volume.
    volume.commit()