### sushi-sashimi-cnn

# Sushi vs Sashimi Image Classification CNN Model
![image](https://github.com/gordon801/my-first-nn/assets/62014067/8509f75f-b2b1-4a19-9b1c-763ec996eb07)

## Overview

This repository contains code and resources for building a Convolutional Neural Network (CNN) model to classify images of Sushi and Sashimi using the FastAI deep learning library. The goal of this project is to create a model that can accurately distinguish between these two types of dishes commonly found in Japanese cuisine.

## Project Highlights

- **Dataset**: FASTAI's FOOD dataset was used. It contains labeled images with various examples of Sushi and Sashimi.
- **Model**: The FastAI library was employed to create and train a CNN model, utilizing the transfer learning approach with a pre-trained ResNet-34 architecture.
- **Training**: The model was fine-tuned using the training data with data augmentation and normalization techniques.
- **Evaluation**: The model's performance was assessed using metrics such as classification accuracy and error rate.
- **Inference**: The trained model can make predictions on new images to classify whether they are Sushi or Sashimi.

## Getting Started

1. Clone this repository:
```
git clone https://github.com/gordon801/sushi-sashimi-classification.git
```
2. Install the required dependencies:
```
conda env update -n my_env --file environment.yml
```
or
```
conda env create -f environment.yml
```
3. Train the model by running the notebook `sushi-vs-sashimi-cnn.ipynb`.

4. Make predictions on new images using the trained model. 
```
python predict.py path/to/image.jpg
```

## Results
The trained model achieved an accuracy of approximately 92% on the validation dataset, demonstrating its effectiveness in classifying Sushi and Sashimi images.

## Example Predictions
![sashimi_test1](https://github.com/gordon801/my-first-nn/assets/62014067/ea0ea642-131f-4a00-9a2b-cfa4e9260c16)
```
$ python predict.py test_data/sashimi_test1.jpg
Prediction: sashimi
Probability: 100.00%
```
![sashimi_test2](https://github.com/gordon801/my-first-nn/assets/62014067/36812f0d-12bf-464e-a2f5-7f341ce0a0fd)
```
$ python predict.py test_data/sashimi_test2.jpg
Prediction: sashimi
Probability: 99.86%
```

![sushi_test1](https://github.com/gordon801/my-first-nn/assets/62014067/96145b25-d827-4857-84a7-25bb8dc2f394)
```
$ python predict.py test_data/sushi_test1.jpg
Prediction: sushi
Probability: 100.00%
```

![sushi_test2](https://github.com/gordon801/my-first-nn/assets/62014067/18e2fb81-914e-4880-8223-0ff98a651154)
```
$ python predict.py test_data/sushi_test2.jpg
Prediction: sushi
Probability: 99.81%
```

## Acknowledgments
- [FastAI Library](https://docs.fast.ai/)
- [Sushi vs Sashimi Image](https://thisonevsthatone.com/sushi-vs-sashimi/)
- [Pytorch Tutorial](https://youtu.be/k1GIEkzQ8qc?si=FEv_pFYHeuvBkStW)


