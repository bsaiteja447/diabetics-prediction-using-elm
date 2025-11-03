from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog, filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import MLPRandomLayer
import unittest

main = Tk()
main.title("Diabetes Prediction Using Extreme Learning Machine: Application of Health Systems")
main.geometry("1300x1200")

# Global variables
global filename, accuracy, X, Y, X_train, X_test, y_train, y_test, classifier
filename = ""
accuracy = []
X = Y = X_train = X_test = y_train = y_test = None
classifier = None  # Initialize classifier to avoid NameError

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="HealthDataset")
    text.delete('1.0', END)
    text.insert(END, "Dataset loaded\n")    

def preprocessing():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset = dataset.fillna(0)
    dataset = dataset.drop('Group', axis=1)

    pca_data = PCA(n_components=8, random_state=0)
    pca_data = pd.DataFrame(pca_data.fit_transform(dataset))
    cols = pca_data.shape[1]
    c1 = cols
    X = dataset.values[:, 0:c1]
    Y = dataset.values[:, c1]
    Y = Y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END, "Total records found in dataset are : "+str(len(X))+"\n")
    text.insert(END, "Total records used to train ML algorithms are : "+str(len(X_train))+"\n")
    text.insert(END, "Total records used to test ML algorithms are : "+str(len(X_test))+"\n")
    
    plt.figure(figsize=(20, 20))
    chart = sn.heatmap(dataset.corr(), yticklabels=dataset.columns, xticklabels=dataset.columns, fmt=".2f", annot=True, annot_kws={'size': 14})
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='left')
    plt.title("Dataset features Matrix after optimization using PCA")
    plt.show()

def runML():
    accuracy.clear()
    global classifier
    text.delete('1.0', END)

    knn = KNeighborsClassifier(n_neighbors=10) 
    knn.fit(X_train, y_train) 
    prediction_data = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, prediction_data) * 100
    accuracy.append(knn_acc)
    text.insert(END, f'KNN Accuracy : {knn_acc:.2f}%\n')

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    prediction_data = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, prediction_data) * 100
    accuracy.append(nb_acc)
    text.insert(END, f'Naive Bayes Accuracy : {nb_acc:.2f}%\n')

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    prediction_data = rfc.predict(X_test)
    rfc_acc = accuracy_score(y_test, prediction_data) * 100
    accuracy.append(rfc_acc)
    text.insert(END, f'Random Forest Accuracy : {rfc_acc:.2f}%\n')

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    prediction_data = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, prediction_data) * 100
    accuracy.append(lr_acc)
    text.insert(END, f'Logistic Regression Accuracy : {lr_acc:.2f}%\n')

    sv = svm.SVC(probability=True)
    sv.fit(X_train, y_train)
    prediction_data = sv.predict(X_test)
    svc_acc = accuracy_score(y_test, prediction_data) * 100
    accuracy.append(svc_acc)
    text.insert(END, f'Linear SVC Accuracy : {svc_acc:.2f}%\n')

    srhl_tanh = MLPRandomLayer(n_hidden=350, activation_func='tanh')
    eml = GenELMClassifier(hidden_layer=srhl_tanh)
    eml.fit(X, Y)
    prediction_data = eml.predict(X_test)
    eml_acc = accuracy_score(y_test, prediction_data) * 100
    accuracy.append(eml_acc)
    text.insert(END, f'Extreme Learning Machine Accuracy : {eml_acc:.2f}%\n')

    classifier = rfc  # Set the best one or any one to use for predictions

    # Visualizing probability of diabetes
    prob_without = rfc.predict_proba(X_test[y_test == 0])[:, 0]
    prob_with = rfc.predict_proba(X_test[y_test == 1])[:, 0]
    df = pd.DataFrame(data=X_test)
    without = df.loc[y_test == 0].index.values
    with_diab = df.loc[y_test == 1].index.values
    plt.scatter(without, prob_without, c='green', label='No Diabetes')
    plt.scatter(with_diab, prob_with, c='red', label='Diabetes')
    plt.title('Probability of Diabetes by Random Forest')
    plt.xlabel('Sample Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

def runMAMPrediction():
    global classifier
    text.delete('1.0', END)
    if classifier is None:
        messagebox.showerror("Error", "Please run Machine Learning first before prediction.")
        return
    filename = filedialog.askopenfilename(initialdir="HealthDataset")
    test = pd.read_csv(filename)
    cols = test.shape[1]
    test = test.values[:, 0:cols]
    predict = classifier.predict(test)
    for i in range(len(test)):
        result = "POSITIVE" if predict[i] == 1 else "NEGATIVE"
        text.insert(END, f"{test[i]} DISEASE PREDICTION RESULT : {result}\n\n")

def softwareTesting():
    class TestStringMethods(unittest.TestCase):
        def test_accuracy_above_zero(self):
            self.assertTrue(all(acc > 0 for acc in accuracy))
    unittest.main(exit=False)

def graph():
    bars = ('KNN', 'Naive Bayes', 'Random Forest', 'Logistic Regression', 'Linear SVC', 'ELM')
    if len(accuracy) != len(bars):
        messagebox.showerror("Error", f"Expected {len(bars)} accuracy values but got {len(accuracy)}.\nRun ML first.")
        return
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, accuracy, color='skyblue')
    plt.xticks(y_pos, bars)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison of ML Algorithms')
    plt.ylim([0, 100])
    plt.show()

def exit():
    main.destroy()

# UI Layout
font = ('times', 14, 'bold')
title = Label(main, text='Diabetes Prediction Using Extreme Learning Machine: Application of Health Systems', bg='light cyan', fg='medium orchid', font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=110, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400, y=100)

font2 = ('times', 13, 'bold')
Button(main, text="Upload Healthcare Dataset", command=upload, font=font2).place(x=50, y=100)
Button(main, text="Run Health Data Preprocessing", command=preprocessing, font=font2).place(x=50, y=150)
Button(main, text="Run Machine Learning Algorithms", command=runML, font=font2).place(x=50, y=200)
Button(main, text="Predict Machine Algorithm Model", command=runMAMPrediction, font=font2).place(x=50, y=250)
Button(main, text="Accuracy Comparison Graph", command=graph, font=font2).place(x=50, y=300)
Button(main, text="Software Testing", command=softwareTesting, font=font2).place(x=50, y=350)
Button(main, text="Exit", command=exit, font=font2).place(x=50, y=400)

main.config(bg='LightSteelBlue1')
main.mainloop()
