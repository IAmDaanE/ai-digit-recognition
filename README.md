# MNIST Digit Recognition

A neural network algorithm that recognizes and classifies hand-drawn digits. Written in Python using my very own [barebones_ml](https://github.com/IAmDaanE/bare-bones-ml) neural network & machine learning library.

---

<img width="743" height="590" alt="MNIST Drawing Environment" src="https://github.com/user-attachments/assets/55823e0d-9652-4b92-af39-814db965db86" />

*The interactive drawing environment*

## About the Project

This repository contains a Python script to train the image classification algorithm, two pretrained models, and a GUI environment where you can draw a digit and see the model's prediction in real-time. 

I used the open-source **MNIST dataset**, which contains 70,000 labeled grayscale images (60,000 for training and 10,000 for testing). To make downloading and loading the data easier, I used the CSV format available [on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

I already pretrained 2 models which you can find in the models/ folder, more info on them in models/README.md.

## Project Structure

```text
ml-digit-classification/
├── data/                  # Directory for the MNIST dataset CSV files
├── models/                # Pretrained models stored as pure NumPy arrays (weights and biases)
└── src/
    ├── accuracy_tester.py # Tests model accuracy on the test dataset
    ├── predict_drawing.py # The GUI environment for drawing digits
    ├── train.py           # The main training loop
    └── utils.py           # Reusable helper functions
```

## Getting Started

### Installing Libraries

**Requires:** Python 3.10 - 3.14

```
pip install -r requirements.txt
```

### Running the Drawing Environment

To test the pretrained model yourself, run the following command in the project root:
```bash
python src/predict_drawing.py "../models/v2" 
```
Want to use a different model? Replace the string with the path to the desired one.

### Training Your Own Model

Because the dataset files are too large for a GitHub repository, you will need to download `mnist_train.csv` and `mnist_test.csv` manually from [this Kaggle page](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

1. Download the CSV files and place them inside the `data/` directory.

2. Open `src/train.py` and configure your desired hyperparameters.

3. Start the training process, replace the string with the location of where you want to stored the trained model's weights and biases:

   ```bash
   python src/train.py "../models/v3"
   ```

4. When satisfied with the loss convergence during training, press `Ctrl + C` to safely stop training and save the trained weights and biases.

### Testing a Models's Accuracy

To measure the accuracy of a model run the following command in the project root:
```bash
python src/accuracy_tester.py "../models/v2" 
```
Want to test a different model? Replace the string with the path to the desired one.

## License

This project is open-source and available under the **MIT License**.