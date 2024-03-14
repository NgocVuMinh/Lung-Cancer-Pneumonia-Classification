# The application of deep learning in lung cancerous lesion detection

### Contributors: [Ngoc M. Vu](https://github.com/NgocVuMinh), [Phuong T.M. Chu](https://github.com/ReiCHU31), [Tram P.B. Ha](https://github.com/nhokchihiro)

This repository contains scripts used to fine-tune deep learning models for the classification of lung cancer and non-cancer pneumonia-only using chest CT scans. Gradient-weighted Class Activation Mapping (GradCAM) was also employed to interpret the models' decision in classifying cancerous CT scans.

![](https://github.com/NgocVuMinh/Lung-Cancer-Pneumonia-Classification/blob/main/overview1.png)

## Model architectures

Nine model architectures were optimized, trained and evaluated in our study:
* DenseNet121
* MobileNetV2
* InceptionV3
* InceptionResNetV2
* ResNet50
* ResNet101
* VGG16
* VGG19
* Xception

All models were built using TensorFlow's Keras API 2.13 with python 3.8.

## Datasets

* For the lung cancer dataset, chest CT scans of 101 lung cancer patients was randomly obtained from the Lung-PET-CT-Dx database in the [Cancer Image Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216). The original database contains approximately 251,135 scans of 355 lung cancer patients.

* A collection of pneumonia-only CT scans was employed as the non-cancer dataset. This is a newly established library extracted from peer-reviewed scientific publications and is available at [ReiCHU31/CT-pneumonia-dataset](https://github.com/ReiCHU31/CT-pneumonia-dataset).

Here, we provided some sample data for training, validation and testing. The `./sample_data` directory is organized as follows:
```
sample_data
├── Training
│   ├── Cancer       
│   └── Non-cancer    
├── Validation
│   ├── Cancer       
│   └── Non-cancer             
├── Testing
│   ├── Cancer       
│   └── Non-cancer       
```

## Usage

* Install packages:
```
pip install -r requirements.txt
```

* Training and testing:

Choose one of the following for the `--base` argument: 
`dense` `mob` `incepres` `res50` `res101` `incepv3` `vgg16` `vgg19` `xcep`

Example command:
```
python main.py --data ./sample_data --base incepv3 --input_size 299 --epochs 5 --batch 16
```

An example notebook to train InceptionV3 was also provided: [example_notebook.ipynb](https://github.com/NgocVuMinh/Lung-Cancer-Pneumonia-Classification/blob/main/example_notebook.ipynb)

* Implementing GradCAM on a chosen input image (remember to specify the path to your saved weights and input image):
```
python gradcam.py --base incepv3 --input_size 299 --model_path "incepv3.h5" \
        --layer mixed10 --img "./gradcam_cancer_input.png"
```
Example GradCAM output:
<p float="left">
  <img src="https://github.com/NgocVuMinh/Lung-Cancer-Pneumonia-Classification/blob/main/gradcam_cancer_input.png" width="100" />
  <img src="https://github.com/NgocVuMinh/Lung-Cancer-Pneumonia-Classification/blob/main/gradcam_output.png" width="100" /> 
</p>