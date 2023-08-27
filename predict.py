import sys
from fastai.vision.all import *

def GetLabel(fileName):
    # Function to get label of image
    return fileName.split('_')[0]

def MakePredictionOnData(imagePath, learnerModel):
    # Load and preprocess image from the dataset
    dataset_image = PILImage.create(imagePath)
    preprocessed_dataset_image = learnerModel.dls.valid.after_batch(dataset_image)

    # Make prediction using our trained model
    dataset_prediction, dataset_label, dataset_probs = learnerModel.predict(preprocessed_dataset_image)
    print(f"Prediction: {dataset_prediction}")
    print(f"Probability: {dataset_probs[dataset_label]:.2%}")
    dataset_image.show()

def main():
    if len(sys.argv) != 2:
        print("Expected input: python predict.py path/to/image.jpg")
        return

    image_path = sys.argv[1]
    model_path = 'model.pkl' 

    # Import model
    learnImported = load_learner(model_path)

    # Make prediction on image
    MakePredictionOnData(image_path, learnImported)

if __name__ == "__main__":
    main()