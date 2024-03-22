# Design Document: Ensemble Modeling of Multiple Physical Indicators to Dynamically Phenotype Autism

This project aims at delivering a binary classification of ASD (ASD: 1, Neurotypical (NT) : 0) using **eye gazing**, **head pose** and **facial landmarks**. 

We first provide individual models for each physical markers (ie. predicting ASD using respectively eye gazing, head pose and facial landmarks). We then constructed an ensemble model using the three markers. 

Our models were trained on videos from the GuessWhat mobile application, designed by the Wall Lab. 

This project was our final project of the class BMI212 at Stanford for the year 2022-2023. 

## Results
We first preprocessed and filtered our data to construct a high-quality, structured and information rich dataset for our project. We reached a test accuracy of 60% using eye gazing, 57% using facial landmarks and 62% using head pose. We then designed an ensemble model able to reach a test accuracy of 69% using the three physical markers. 

## Installing the requirements
To install the requirements, run the command `conda env create -f environment.yml`. This will create and install the conda environment `bmi212` with all needed requirements. 

## Running the code
You can train individual models by running the script `train.py` in the folder `individual_models`. You can call this script with many parameters to choose what you want to do. Here is a quick overview of the most important parameters : 
- `--feature [OPTION]`, with `[OPTION] = eye, head, facial`. Choose what feature you want to use to predict ASD or NT.
- `--model [OPTION`, with `[OPTION] = lstm, gru, modifiedgru, modifiedlstm`. Choose what model you want to use to predict ASD or NT.
- Some hyperparameters: `--learning_rate [LR]`, `--batch_size [BS]`, `--num_epochs [EPOCHS]`, `--hidden_size [HIDDEN SIZE]`, `--num_layers [NUM LAYERS]`, `--dropout [DROPOUT]` to run the code on GPU and use AMP (Automated Mixed Precision) in order to have the best performances.
- `--tune_hyperparameters` to tune the hyperparameters using the framework Optuna. (This will run by default 10 trials in the defined hyperparameters search space. 

## Example of how to train a working model for eye gazing

You can train a working model for eye gazing by running the following command : 

`python train.py --feature eye learning_rate 0.0017 --num_epochs 11 --num_layers 4 --model modifiedgru --batch_size 25 --dropout 0.25`

## Ensemble
You can train and tune hypereparameters for both late fusion and intermediate fusion models by running: 'python ensemble_train.py'. This will run 50 trials each searching for optimal validation loss.
