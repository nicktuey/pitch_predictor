# Pitch Predictor Neural Network

## Overview

A deep learning model that predicts whether a pitcher will throw a **Fastball** or **Off-Speed** pitch based on game situation context. The model uses a PyTorch neural network to analyze pitch sequencing patterns and provide real-time predictions for strategic decision-making.

## What It Predicts

- **Binary Classification**: Fastball (1) vs. Other Pitch Types (0)
- **Output**: Probability score (0.0 - 1.0) indicating likelihood of fastball
- **Threshold**: 0.5 probability determines final prediction

## Key Features

The model analyzes these game situation variables:
- **Count**: Balls (0-3) and Strikes (0-2)
- **Batter Side**: Left-handed or Right-handed
- **Previous Pitch Type**: Fastball (FB), Breaking Ball (BB), or Off-Speed (OS)
- **Previous Pitch Outcome**: Ball Called, Strike Called, Strike Swinging, or Foul Ball

## Model Architecture

### Neural Network Structure
```
Input Layer: Variable size (based on feature count)
    ↓
Hidden Layer 1: 16 neurons + ReLU activation
    ↓
Hidden Layer 2: 8 neurons + ReLU activation
    ↓
Output Layer: 1 neuron + Sigmoid activation
```

### Technical Specifications
- **Framework**: PyTorch
- **Loss Function**: Binary Cross Entropy (BCELoss)
- **Optimizer**: Adam with learning rate 0.05
- **Training Epochs**: 100
- **Data Split**: 70% train, 15% validation, 15% test

## Data Pipeline

### 1. Data Collection (`dataclean.ipynb`)
- Connects to Private NCAA Trackman database
- Extracts pitcher-specific data (currently configured for Kyle McCoy)
- Features: Pitcher, TaggedPitchType, PitchCall, Balls, Strikes, RelSpeed, PitcherThrows, BatterSide

### 2. Data Preprocessing
- Converts categorical variables to dummy variables
- Handles missing values and data quality issues
- Creates binary target variable (Fastball = 1, Other/OS = 0)

### 3. Model Training (`predictor.ipynb`)
- Trains neural network with cross-validation
- Performs multiple training runs with different random seeds for robustness
- Tracks training/validation/test performance metrics

### 4. Prediction Generation
- Creates comprehensive lookup table for all possible game scenarios
- Generates fastball probability predictions for strategic analysis
- Exports results to CSV for practical application


### Prerequisites

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn mysql-connector-python python-dotenv
```


## Performance

- Performance varies based on pitcher tendencies and data quality*

## Output Files

- **`clean_data.csv`**: Preprocessed training dataset
- **`Predictor_csvs/Maryland_mccoy.csv`**: Comprehensive prediction lookup table for every count and previous pitch group situation. For this example I was using Maryland pitcher Kyle McCoy


## Future Enhancements

- **LTSM Model**: Create an LTSM to create more accurate predictions based on longer sequences of pitches
- **Simplified Previous Pitch**: Predict using two pitch groups (FB and OS which group breaking balls into offspeed pitches) similarly to how the output is either FB or OS



