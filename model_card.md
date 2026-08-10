# Model Card

## Model Details
This is a Random Forest classification model trained to predict whether an 
individual's income is greater than $50,000 a year. 
The model uses Census Bureau demographic and employment data, 
and was trained using categorical feature encoding and numerical features 
from the Census Income dataset.

## Intended Use
This model uses demographic and employment data from the Census Income dataset 
to predict whether an individual's income is greater than $50,000. 

## Training Data
The model was trained using the Census Income dataset.
The dataset contains demographic and employment information, 
including categorical and numerical features.
The data was split into training and test sets, with 80% used for training.
Categorical features were one-hot encoded before training the model.

## Evaluation Data
20% of the Census Income dataset was set aside for testing, 
ensuring the evaluation data was not used for training the model. 
The same preprocessing used on the training data was applied to the evaluation data
using the encoders fitted during training.

## Metrics
The model was evaluated using precision, recall, and F1 score. 
The model achieved a precision of 0.7446, 
a recall of 0.6327, 
and an F1 score of 0.6841 on the evaluation data.

## Ethical Considerations
The model was trained using demographic information such as race and sex, 
which could potentially contribute to bias in the model's predictions. 
Model performance may vary between different groups within the data. 
These factors should be considered when interpreting the model's predictions.

## Caveats and Recommendations
The model's performance may vary depending on the characteristics of the data being used. 
The model should be evaluated on additional data and adjusted as needed before being used for real-world applications. 
Consider evaluating the model across various demographic groups to identify potential biases in its performance.

