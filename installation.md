<h1>Installation Steps for Object Detection API for TF2 </h1>
Steps: <br>
1. Open anaconda powershell prompt (you can window+s and search for it), key in the following commands: <br>

```bash
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

<br><br>
<h1>Installation Steps for Darkflow </h1>

```bash
git clone https://github.com/thtrieu/darkflow.git
conda create -n dkf_env python=3.5
###
python.exe -m pip install pip==20.3
pip install tensorflow==1.0
conda install cython
###
# Follow the steps below to install ms C++ build tools
conda activate dkf_env
python setup.py build_ext --inplace
```

<h3>Step to install MS C++ build tools: </h3>
1. Go to this <a href="https://visualstudio.microsoft.com/visual-cpp-build-tools/"> link</a> to download. <br>
2. Click 'Modify', and under the 'Optional', tick 'MSVC v140 ...' then download it. <br>
3. After downloading, launch the command prompt, then cd to your working folder.
