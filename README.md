# CDS6334-VIP-Project
This is the official repository for VISUAL INFO PROCESSING Project Trimester 2430
<br>
This project is contributed by:
| Member | Name | ID 
| -------- | ------- | ------- |
| 1 | ANG KHAI PIN | 1211101248 |
| 2 | AHMAD DANIAL BIN AHMAD FAUZI | 1211100824 |
| 3 | JAVIER AUSTIN ANAK JAWA | 1211100857 |
| 4 | MUHAMMAD ZAFRAN BIN MOHD ANUAR | 1211101321 |

## Disclaimer
This project is not an original work. This is an implementation work from <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API </a> and the research paper: <a href="https://www.sciencedirect.com/science/article/abs/pii/S092523121830924X?via%3Dihub">Evaluation of deep neural networks for traffic sign detection systems</a>.

## Prerequisite
1. Make sure <a href="https://docs.anaconda.com/anaconda/install/">Anaconda</a> is installed.
2. Head over to this <a href="https://drive.google.com/drive/folders/14XZOY5WXXT9TOM_oIbsj7Qd92L1Os-5g"> g.drive link </a> to download the dataset, pre-trained models and nvvm folder.
3. Unzip the dataset into `./mtsd_data/(unzip here)`.
4. Unzip the models into `./models/(unzip here)`.
5. (In case you wants to fine-tune the model) Unzip the nvmm folder into the *cloned tensorflow object detection* folder: `./models/research/(unzip here)`.

## Installation Steps
<b>Installation Steps for Object Detection API for TF2</b>

1. Open anaconda powershell prompt (you can window+s and search for it), and key in the following commands 1 by 1:
```bash
cd C:/TF Obj Det/ #to your prefered file location
git clone https://github.com/tensorflow/models.git
cd models/research
###
conda create -n oda-tf2_env python=3.7
conda activate oda-tf2_env
conda install protobuf
###
protoc object_detection/protos/*.proto --python_out=.
###
cp object_detection/packages/tf2/setup.py .
python.exe -m pip install pip==20.2
###
python -m pip install --use-feature=2020-resolver .
###
pip install urllib3==1.26.6
python object_detection/builders/model_builder_tf2_test.py
```
2. The last line is to test run the tensorflow model, if it returns 24 successful tests, you will be good to go to run this repo.

## Instructions to run the fine-tuned model on new images
1. The .py file to test run the model is <a href="./scripts/model_test_run.py">here</a>.
2. Change the variable `model_name` to point to the model you want to test run.
3. Change the variable `img` to the images you want to test on.

## Instructions for fine-tuning your model
1. Install Tensorflow GPU (only available for NVIDIA GPUs). In your command prompt:
``` bash
conda activate oda-tf2_env
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
# checking to make sure cuda is installed
nvcc --version
pip install tensorflow-gpu==2.10.0
```
2. Modify the `pipeline.config` file in the `./models/ssd.../pipeline.config` to suit your dataset and requirements.
3. Open command prompt and type in the commands from <a href="commands.txt"> commands.txt</a>. Be sure to change the filepath of `--pipeline_config_path`, `--model_dir`, `--checkpoint_dir`, `--trained_checkpoint_dir` and `--output_directory` in the commands.txt.

## Evaluation
1. Run the <a href="scripts/run_inference.py">inference</a> python file to create Inference Model. Then head to <a href="Evaluation.ipynb">evaluation</a> to evaluate the model.

## Acknowledgements
The source code of this project is mainly based on <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API </a> and the research paper: <a href="https://www.sciencedirect.com/science/article/abs/pii/S092523121830924X?via%3Dihub">Evaluation of deep neural networks for traffic sign detection systems</a>.
