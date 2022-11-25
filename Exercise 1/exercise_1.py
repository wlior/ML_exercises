import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics

"""Question 1 - Bayes' Classifier"""

print("--- Question 1 ---\n")
# Probabilities for coin results - variable Y
coin_probs = {
    0: 0.8,
    1: 0.2}


dice_probs = {
# Probabilities for die #0 (rolled if Y == 0)
    0: {
        1: 0.1,
        2: 0.2,
        3: 0.1,
        4: 0.2,
        5: 0.2,
        6: 0.2
    },
# Probabilities for die #1 (rolled if Y == 1)
    1: {
        1: 0.2,
        2: 0.2,
        3: 0.2,
        4: 0.1,
        5: 0.1,
        6: 0.2
    }
}

def bayes_classifier(dice_value: int):
    # Calculate the enumerator for Bayes' theorem 
    enumerator = dice_probs[1][dice_value] * coin_probs[1]
    # Calculate the denominator for Bayes' theorem
    denominator = 0
    for i in (0,1):
        denominator += coin_probs[i] * dice_probs[i][dice_value]
    
    probability_of_heads = enumerator / denominator

    if probability_of_heads > 0.5:
        result = "The likely result is heads"
    elif probability_of_heads < 0.5:
        result = "The likely result is tails"
    else:
        result = "Heads and Tails are equally likely"
    
    print(f"The probability to get heads if the dice roll was {dice_value} is {round(probability_of_heads,4)}, and therefore {result}.")
    return result

# Print the prediction for each result of the dice
for i in range(1,7):
    bayes_classifier(i)
print ("--- End Question 1 ---\n")



"""Question 2: Specificity, Sensitivity and Bayes' rule"""

print("--- Question 2 ---\n")
# Sensitivity is the probability to get a positive test result if the patient is really positive (AKA P(X=1|Y=1))
sensitivity = 0.9

# Specificity is the probability to get a negative test result if the patient is really negative (AKA P(X=0|Y=0))
specificity = 0.98

# Calculate the conditional probability based on Bayes' theorem and the definitions of spcificity and sensitivity
def prob_true_positive(sensitivity: float, specificity: float, disease_prevalence: float):
    probability_TP = sensitivity * disease_prevalence / ((sensitivity * disease_prevalence) + ((1 - specificity) * (1 - disease_prevalence)))
    print(f"If the test the result is positive and the disease prevalence is {disease_prevalence}, there is a {round(100*probability_TP,3)}% chance that the patient is truly positive.")

prevalence = 0.1
prob_true_positive(sensitivity, specificity, prevalence)

prevalence = 0.02
prob_true_positive(sensitivity, specificity, prevalence)

print("--- End Question 2 ---\n")


"""Question 3 - Generalized least squares"""
print ("--- Question 3 ---\n")
# Append intercept 
def intercept_appender(x: np.ndarray):
    intercept = np.ones((len(x), 1))
    return np.hstack([intercept,x])


def coefficient_calculator(y: np.ndarray ,x: np.ndarray, n: np.ndarray ,sigma_squared: float):
    # Calculate weight matrix
    W_matrix = (1 / sigma_squared) * np.diag(n)
    # Add intercept column to x matrix
    x = intercept_appender(x)
    # Calculation of parameter estimation based on given calculation. Result is squeezed for aesthetic purposes.
    return np.squeeze(np.linalg.inv(x.T @ W_matrix @ x) @ (x.T @ W_matrix @ y))

# Test example:
y = np.array([[140],[148],[155],[149],[160]])
x = np.array([[40],[43],[47],[44.5],[54]])
n = np.array([31,36,34,39,33])
sigma = 200
# print(x,y,n,sigma)
betas = coefficient_calculator(y,x,n,sigma)
print(f"The intercept is {round(betas[0],3)} and beta_1 is {round(betas[1],3)}.")


print("--- End Question 3 ---\n")



"""Question 4 - Logistic regression"""

print ("--- Question 4 ---\n")
# Section A: data preparation

URL = "https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0234552.s004"
df = pd.read_csv(URL, index_col=0)

df = df[['BMI', 'Waist circumference', 'Height_value-1', 'Visc']]
df['WHtR'] = df['Waist circumference'] / df['Height_value-1']

visc_median = df['Visc'].median()

df['Visc_bool'] = (df['Visc'] > visc_median).astype(int)


# Train-test split preparation
feature_names = ['BMI', 'Waist circumference', 'WHtR']
target_name = 'Visc_bool'
X = df[feature_names]
Y = df[target_name].values
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=0, test_size=0.2)

# Create logistic regression model
our_model = sklearn.linear_model.LogisticRegression(random_state=0, penalty="none", solver="newton-cg")
_ = our_model.fit(X_train, Y_train)

# Predicted values for test set

predictions = our_model.predict(X_test)

# Section C: Calculate The AUC of the ROC for our model
AUC = sklearn.metrics.roc_auc_score(Y_test, predictions)
print(f"The Area under the ROC curve for this model is {AUC}")

# Section D: 0.4 threshold instead of 0.5, printing confusion matrix, sensitivity, specificity, accuracy

def predict_forty_percent(predictors):
    probabilities = our_model.predict_proba(predictors)
    return (probabilities[:, 1] > 0.4).astype(int)

new_predictions = predict_forty_percent(X_test)
#print(new_predictions)

# Confustion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(Y_test, new_predictions)
print(f"The confusion matrix for this model is:\n {confusion_matrix}")

# Pretty confustion matrix - how do we see the printed matrix?
pretty_confusion_matrix = sklearn.metrics.plot_confusion_matrix(our_model, X_test, Y_test)
_ = pretty_confusion_matrix.ax_.set_title(f"Confusion Matrix")

sensitivity_2 = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
specificity_2 = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix)
print(f"The sensitivity is {round(sensitivity_2,2)}")
print(f"The specificity is {round(specificity_2,2)}")
print(f"The model accuracy is {round(accuracy,2)}")

print("--- End Question 4 ---")
