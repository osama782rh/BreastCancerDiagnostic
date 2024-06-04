from imports import *

def load_and_analyze_data():
    """
    Load and perform a basic analysis on the breast cancer dataset.
    The function loads the dataset, prints out some basic statistics,
    and returns the features and target variable.
    Returns:
    x (DataFrame): Features of the dataset.
    y (DataFrame): Target variable of the dataset.
    """
    # Load the dataset
    samples = load_breast_cancer()
    x = pd.DataFrame(samples.data, columns=samples.feature_names)
    y = pd.DataFrame(samples.target, columns=["Target"])
    # Basic analysis
    print("Data Loaded and Analyzed")
    print("Number of samples:", x.shape[0])
    print("Number of features:", x.shape[1])
    print("Number of benign samples:", (y['Target'] == 1).sum())
    print("Number of malignant samples:", (y['Target'] == 0).sum())
    return x, y

def prepare_data(x, y):
    """
    Prepares the data for model training by splitting the data into training and test sets
    and then standardizing it.
    Parameters:
    x (DataFrame): Features of the dataset.
    y (DataFrame): Target variable of the dataset.
    Returns:
    x_train_scaled (ndarray): Scaled features for the training set.
    x_test_scaled (ndarray): Scaled features for the test set.
    y_train (DataFrame): Target variable for the training set.
    y_test (DataFrame): Target variable for the test set.
    """
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # Standardizing the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("Data Prepared")
    return x_train_scaled, x_test_scaled, y_train, y_test

def train_models(x_train_scaled, y_train):
    """
    Trains Logistic Regression and Support Vector Machine models on the training data.
    Parameters:
    x_train_scaled (ndarray): Scaled features for the training set.
    y_train (DataFrame): Target variable for the training set.
    Returns:
    log_reg (LogisticRegression): Trained Logistic Regression model.
    svm_model (SVC): Trained Support Vector Machine model.
    """
    # Training Logistic Regression Model
    log_reg = LogisticRegression()
    log_reg.fit(x_train_scaled, y_train.values.ravel())
    # Training Support Vector Machine Model
    svm_model = SVC()
    svm_model.fit(x_train_scaled, y_train.values.ravel())
    print("Models Trained")
    return log_reg, svm_model

def evaluate_models(log_reg, svm_model, x_test_scaled, y_test):
    """
    Makes predictions using the trained models and evaluates their performance.
    Parameters:
    log_reg (LogisticRegression): Trained Logistic Regression model.
    svm_model (SVC): Trained Support Vector Machine model.
    x_test_scaled (ndarray): Scaled features for the test set.
    y_test (DataFrame): Target variable for the test set.
    """
    # Making predictions
    log_reg_pred = log_reg.predict(x_test_scaled)
    svm_pred = svm_model.predict(x_test_scaled)
    # Evaluating the models
    log_reg_report = classification_report(y_test, log_reg_pred)
    svm_report = classification_report(y_test, svm_pred)
    print("Logistic Regression Report:\n", log_reg_report)
    print("SVM Report:\n", svm_report)
    # Plotting and displaying confusion matrices
    log_reg_cm = confusion_matrix(y_test, log_reg_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(log_reg_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Logistic Regression Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.subplot(1, 2, 2)
    sns.heatmap(svm_cm, annot=True, fmt="d", cmap="YlGn")
    plt.title("SVM Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    """
    Main function to run the breast cancer data analysis pipeline.
    Provides a menu for the user to choose from loading data, preparing data,
    training models, evaluating models, or exiting the program.
    """
    x, y, x_train_scaled, x_test_scaled, y_train, y_test, log_reg, svm_model = None, None, None, None, None, None, None, None
    while True:
        print("\nMenu:")
        print("1. Load and Analyze Data")
        print("2. Prepare Data")
        print("3. Train Models")
        print("4. Evaluate Models")
        print("5. Exit")
        choice = input("Enter your choice: ")
    if choice == '1':
        x, y = load_and_analyze_data()
    elif choice == '2':
        x_train_scaled, x_test_scaled, y_train, y_test = prepare_data(x, y)
    elif choice == '3':
        log_reg, svm_model = train_models(x_train_scaled, y_train)
    elif choice == '4':
        evaluate_models(log_reg, svm_model, x_test_scaled, y_test)
    elif choice == '5':
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please choose a valid option.")


if __name__ == "__main__":
    main()