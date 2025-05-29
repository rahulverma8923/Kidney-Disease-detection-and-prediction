import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 


class Model:

    def __init__(self):
        self.name = ''
        path = 'dataset/kidney_disease.csv'
        df = pd.read_csv(path)
        df = df[['age', 'bp', 'su', 'pc', 'pcc', 'sod', 'hemo', 'htn', 'dm', 'classification']]

        # Fill missing values
        df['age'] = df['age'].fillna(df['age'].mean())
        df['bp'] = df['bp'].fillna(df['bp'].mean())
        df['su'] = df['su'].fillna(df['su'].mode()[0])
        df['pc'] = df['pc'].fillna(df['pc'].mode()[0])
        df['pcc'] = df['pcc'].fillna(df['pcc'].mode()[0])
        df['sod'] = df['sod'].fillna(df['sod'].mode()[0])
        df['hemo'] = df['hemo'].fillna(df['hemo'].mode()[0])
        df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
        df['dm'] = df['dm'].str.replace(" ", "").str.replace("\t", "").fillna(df['dm'].mode()[0])
        df['classification'] = df['classification'].str.replace("\t", "").fillna(df['classification'].mode()[0])

        # Label encoding
        labelencoder = LabelEncoder()
        df['pc'] = labelencoder.fit_transform(df['pc'])
        df['pcc'] = labelencoder.fit_transform(df['pcc'])
        df['htn'] = labelencoder.fit_transform(df['htn'])
        df['dm'] = labelencoder.fit_transform(df['dm'])
        df['classification'] = labelencoder.fit_transform(df['classification'])

        self.split_data(df)

    def split_data(self, df):
        x = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values    # Last column
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=24)

    def svm_classifier(self):
        self.name = 'SVM Classifier'
        classifier = SVC()
        return classifier.fit(self.x_train, self.y_train)

    def decision_tree_classifier(self):
        self.name = 'Decision Tree Classifier'
        classifier = DecisionTreeClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def random_forest_classifier(self):
        self.name = 'Random Forest Classifier'
        classifier = RandomForestClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def naive_bayes_classifier(self):
        self.name = 'Naive Bayes Classifier'
        classifier = GaussianNB()
        return classifier.fit(self.x_train, self.y_train)

    def knn_classifier(self):
        self.name = 'KNN Classifier'
        classifier = KNeighborsClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def accuracy(self, model):
        predictions = model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)  # Use np.sum for total count
        print(f"{self.name} has accuracy of {accuracy * 100:.2f}%")

if __name__ == '__main__':
    model = Model()
    model.accuracy(model.svm_classifier())
    model.accuracy(model.decision_tree_classifier())
    model.accuracy(model.random_forest_classifier())
    model.accuracy(model.naive_bayes_classifier())
    model.accuracy(model.knn_classifier())