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

## Installation Steps
<b>Installation Steps for Object Detection API for TF2</b>

1. Open anaconda powershell prompt (you can window+s and search for it), and key in the following commands 1 by 1: <br>

```bash
cd #to your prefered file location
git clone https://github.com/tensorflow/models.git
cd models/research
###
conda create -n oda-tf2_env python=3.7
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


## Instructions for fine-tuning your model
1. Install Tensorflow GPU (only available for NVIDIA GPUs)


## Instructions for evaluating your model