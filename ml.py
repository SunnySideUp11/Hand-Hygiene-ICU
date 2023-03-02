from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble  import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, RandomizedSearchCV
from sklearn import metrics
import os
import joblib


class myModel:
    classifier = {
        "LR": LogisticRegression(penalty='l2', max_iter=10000),
        "RF": RandomForestClassifier(n_estimators=7),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(126, 256), max_iter=10000, random_state=1),
        "SVC": SVC(kernel='rbf', probability=True)
    }
    
    def __init__(self, model_name):
        assert model_name in self.classifier.keys(), "There is no model for you"
        self.model_name = model_name
        self.model = self.classifier[model_name]
    
    def train(self, data):
        x_train, y_train = data
        self.model.fit(x_train, y_train)
    
    def test(self, data):
        x_test, y_test = data
        y_pred = self.model.predict(x_test)
        results = { 
            "model_name": self.model_name,
            "accuracy_score": metrics.accuracy_score(y_test, y_pred),
            "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
            "f1_score": metrics.f1_score(y_test, y_pred, average="macro"),
            "recall_score": metrics.recall_score(y_test, y_pred, average="macro"),
        }
        return results
    
    def predict(self, X):
        return self.model.predict(X)
    

class Trainer:
    def __init__(self, model, X, Y, mode=None, n_splits=5, n_repeats=3, save_folder="./output"):
        self.model = model
        self.data = (X, Y)
        self.save_folder = os.path.join(save_folder, "generalization")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        assert mode in [None, "KF", "RKF"], "There is no training mode for you"
        if mode == "KF":
            self.training_mode = KFold(n_splits, shuffle=True, random_state=1)
        elif mode == "RKF":
            self.training_mode = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=1)
        else:
            self.training_mode = None
            
        self.init_params = model.model.get_params()
    
    def train_one_split(self, data_train, data_test):
        if self.best_params:  
            self.model.model.set_params(**self.best_params)
        else:
            self.model.model.set_params(**self.init_params)
        self.model.train(data_train)
        return self.model.test(data_test)
        
    def run(self, best_params=None):
        self.best_params = best_params
        OUTPUT = []
        best_acc = 0.0
        X, Y = self.data
        if self.training_mode == None:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)
            result = self.train_one_split((x_train, y_train), (x_test, y_test))
            OUTPUT.append(result)
            acc = result["accuracy_score"]
            if acc > best_acc:
                best_acc = acc
                joblib.dump(self.model, os.path.join(self.save_folder, f"{self.model.model_name}.pkl"))
        else:
            for idx_train, idx_test in self.training_mode.split(X):
                x_train, y_train = X[idx_train], Y[idx_train]
                x_test, y_test = X[idx_test], Y[idx_test]
                result = self.train_one_split((x_train, y_train), (x_test, y_test))
                OUTPUT.append(result)
                acc = result["accuracy_score"]
                if acc > best_acc:
                    best_acc = acc
                    joblib.dump(self.model, os.path.join(self.save_folder, f"{self.model.model_name}.pkl"))
              
        return OUTPUT
    

def find_best_SVC(X, Y):
    """
    Args:
        X (ndarray): data
        Y (ndarray): label

    Returns:
        dict: best parameters
    """
    svc = SVC(kernel="rbf", probability=True)
    random_search = RandomizedSearchCV(
        estimator = svc,
        param_distributions =  {
            "C" : [1e-1, 1, 10, 100],
            "gamma": [1e-2, 1e-1, 1, 10]
        },
        cv = 5,
        scoring = "accuracy"
    )
    result = random_search.fit(X, Y)
    return result.best_params_
    