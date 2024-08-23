# Simple Linear Regression Project

This project implements a simple linear regression model using Python. The model is trained to predict the relationship between the independent variable `x` and the dependent variable `y` using a given training dataset. The model is tested on a separate test dataset, and predictions are made for random input values.

## Project Structure

- `train.csv`: The dataset used to train the model.
- `test.csv`: The dataset used to test the model.
- `linear.py`: Python script that implements the linear regression model, trains it, and makes predictions.

## Features

- Trains a simple linear regression model on a training dataset.
- Makes predictions on the test dataset.
- Computes accuracy metrics such as Mean Squared Error (MSE) and R² Score.
- Visualizes actual vs predicted values using matplotlib.
- Provides the equation of the regression line.
- Predicts values for random inputs.

## Installation

To run this project, you will need to have the following Python libraries installed:

```bash
  pip install numpy pandas scikit-learn matplotlib
```
## Usage
1) Clone the repository:

```bash
  git clone https://github.com/your-username/simple-linear-regression.git
  cd simple-linear-regression
```
2) Add your datasets:
  Place the train.csv and test.csv files in the root directory of the project.

3) Run the script:
  Execute the linear.py script to train the model and generate predictions:

```bash
  python linear.py
```
4) View Results:
The script will print the following:
    - The equation of the regression line.
    - Predicted values for the test dataset.
    - Accuracy metrics: MSE and R² Score.
    - Random predictions for manually inputted values.
    - Additionally, a plot of actual vs predicted values will be displayed.

License
This project is licensed under the MIT License - see the LICENSE file for details.
