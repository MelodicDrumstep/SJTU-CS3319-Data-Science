import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# Function to load training and testing data for a single subject
def load_subject_data(subject_id):
    data_dir = f'../dataset/{subject_id}/'
    X_train = np.load(os.path.join(data_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_label.npy'))
    X_test = np.load(os.path.join(data_dir, 'test_data.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_label.npy'))
    return X_train, y_train, X_test, y_test

# Function to train and evaluate SVM and KNN for a subject
def train_and_evaluate_models(subject_id):
    # Load data for the subject
    X_train, y_train, X_test, y_test = load_subject_data(subject_id)
    
    # SVM model
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f'Subject {subject_id} - SVM Accuracy: {svm_accuracy:.2f}')
    
    # KNN model
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    print(f'Subject {subject_id} - KNN Accuracy: {knn_accuracy:.2f}')
    
    return svm_accuracy, knn_accuracy

# Main function to train models for all subjects
if __name__ == "__main__":
    subjects = [1, 2, 3]  # List of subjects
    results = {}
    
    # Train and evaluate models for each subject
    for subject_id in subjects:
        print(f"\nTraining and evaluating models for Subject {subject_id}...")
        svm_acc, knn_acc = train_and_evaluate_models(subject_id)
        results[subject_id] = {'SVM Accuracy': svm_acc, 'KNN Accuracy': knn_acc}
    
    # Summary of results
    print("\nSummary of results:")
    for subject_id, accuracies in results.items():
        print(f"Subject {subject_id}: SVM Accuracy = {accuracies['SVM Accuracy']:.2f}, KNN Accuracy = {accuracies['KNN Accuracy']:.2f}")
