import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
import signal
from scipy.stats import boxcox, zscore
from pprint import pprint

def handler(signum, frame):
    raise Exception('Function timeout.')

class AutoML(object):
    def __init__(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        
        self.X_means_ = {}
        self.X_modes_ = {}
        
        self.names_ = []
        self.models_ = {}
        self.params_ = {}
        
        self.name_score_ = {}
        self.name_sorted_ = []
    
    def fit(self, name=None):
        if name == None:
            name = self.names_
        
        if isinstance(name, list):
            for nick in name:
                self.fit(nick)
        
        else:
            self.models_[name].fit(self.X_train_, self.y_train_)
    
    def get_params(self, name):
        return self.models_[name].get_params()
    
    def set_predict(self, X, y):
        self.X_predict_ = X
        self.y_predict_ = y
    
    def predict(self, name=None):
        if name == None:
            name = self.name_sorted_[0]
        
        return self.models_[name].predict(self.X_predict_)
    
    def predict_proba(self, name=None):
        if name == None:
            name = self.name_sorted_[0]
        
        return self.models_[name].predict_proba(self.X_predict_)
    
    def score(self, name=None):
        if name == None:
            name = self.name_sorted_[0]
        
        # print self.X_predict_.isnull().sum()
        
        return self.models_[name].score(self.X_predict_, self.y_predict_)
    
    def set_params(self, name, params):
        self.models_[name].set_params(**params)
    
    def add_model(self, name, model, params=None):
        if name not in self.names_:
            self.names_ += [name]
            self.models_[name] = model
            self.params_[name] = params
    
    def add_default_models(self):
#        self.add_model('KNN', KNeighborsClassifier(n_jobs=-1),
#                       {'n_neighbors': [3, 5, 10],
#                        'weights': ['uniform', 'distance']})
        self.add_model('LogReg', LogisticRegression(n_jobs=-1),
                       {'C' : [.1, 1, 10],
                        'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear']})
#        self.add_model('LinearSVM', LinearSVC(),
#                       {'loss': ['hinge', 'squared_hinge']})
        # self.add_model('KernelSVM', SVC(), 
        #                {'kernel': ['linear', 'poly','rbf', 'sigmoid', 'precomputed'], 
        #                 'gamma': [.1, 1, 10]})
#        self.add_model('ExtraTree', ExtraTreeClassifier(),
#                       {'criterion': ['gini', 'entropy'],
#                        'max_features': ['auto', 'sqrt', 'log2']})
#        self.add_model('DecisionTree', DecisionTreeClassifier(),
#                       {'max_features': ['auto', 'sqrt', 'log2'],
#                        'min_samples_leaf': [1, 10, 100]})
#        self.add_model('Bagging', BaggingClassifier(n_jobs=-1),
#                       {'n_estimators': [10, 100],
#                        'max_features': [.2, .5, 1.]})
        self.add_model('RandomForest', RandomForestClassifier(n_jobs=-1),
                       {'n_estimators': [10, 100],
                        'max_features': ['auto', 'sqrt', 'log2']})
#        self.add_model('GradBoost', GradientBoostingClassifier(),
#                       {'n_estimators': [10, 100],
#                        'loss': ['deviance', 'exponential'],
#                        'max_depth': [1, 3, 5]})
#        self.add_model('AdapBoost', AdaBoostClassifier(),
#                       {'n_estimators': [50, 100],
#                        'learning_rate': [.1, .5, 1]})
    
    def search_best_params(self, name=None):
        if name == None:
            name = self.names_
        
        if isinstance(name, list):
            for nick in name:
                self.search_best_params(nick)
        
        elif self.params_[name] != None:
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(60)
                
                try:
                    print 'trying %s with time...' % name,
                    try:
                        adap_search_cv = GridSearchCV(self.models_[name], self.params_[name], cv=7)
                    except Exception as e:
                        adap_search_cv = RandomizedSearchCV(self.models_[name], self.params_[name], n_iter=10, cv=7)
                    
                    adap_search_cv.fit(self.X_train_, self.y_train_)
                    
                    self.models_[name].set_params(**adap_search_cv.best_params_)
                    self.params_[name] = None
                except Exception as e:
                    print 'failed!\ntrying %s without time...' % name,
                    try:
                        adap_search_cv = GridSearchCV(self.models_[name], self.params_[name])
                    except Exception as e:
                        adap_search_cv = RandomizedSearchCV(self.models_[name], self.params_[name], n_iter=1)
                    
                    adap_search_cv.fit(self.X_train_, self.y_train_)
                    
                    self.models_[name].set_params(**adap_search_cv.best_params_)
                    self.params_[name] = None
                
                print 'done!'
                signal.alarm(0)
            except Exception as e:
                try:
                    adap_search_cv = GridSearchCV(self.models_[name], self.params_[name])
                except Exception as e:
                    adap_search_cv = RandomizedSearchCV(self.models_[name], self.params_[name], n_iter=2)
                
                adap_search_cv.fit(self.X_train_, self.y_train_)
                
                self.models_[name].set_params(**adap_search_cv.best_params_)
                self.params_[name] = None
    
    def sort_models(self):
        for name in self.names_:
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(30)
                
                try:
                    cv_score = cross_val_score(self.models_[name], self.X_train_, self.y_train_, cv=7, scoring='accuracy')
                    self.name_score_[name] = cv_score.mean() - cv_score.std()
                except Exception as e:
                    self.name_score_[name] = -1
                
                signal.alarm(0)
            except:
                cv_score = cross_val_score(self.models_[name], self.X_train_, self.y_train_, cv=7, scoring='accuracy')
                self.name_score_[name] = cv_score.mean() - cv_score.std()
            
            left, right = -1, len(self.name_sorted_)
            while right - left > 1:
                middle = (left + right) / 2
                if self.name_score_[name] < self.name_score_[self.name_sorted_[middle]]:
                    left = middle
                else:
                    right = middle
            
            self.name_sorted_ = self.name_sorted_[:left+1] + [name] + self.name_sorted_[left+1:]
    
    def idxPCA(self, L, x):
        temp = 0
        for i in xrange(len(L)):
            temp += L[i]
            if temp > x:
                return range(i+1)
     
    def PrincipalCA(self):
        self.X_train_ = zscore(self.X_train_)
        cov = np.cov(self.X_train_.T)
        ev , eig = np.linalg.eig(cov)
        idx = ev.argsort()[::-1]
        ev = ev[idx]
        eig = eig[:,idx]
        prop = ev/ ev.sum()
        #cum = prop.cumsum()
        eigfix = eig[:, self.idxPCA(prop, .85)]
        self.X_train_= self.X_train_.dot(eigfix)
        #print cum
        
    def cleanse_pref(self):
        row_count = self.X_train_.shape[0]
        column_all = list(self.X_train_.columns)
        
        self.X_trans_ = dict((x, []) for x in column_all)
        
        for col in column_all:
            if float(row_count - self.X_train_[col].isnull().sum()) / row_count < .4:
                self.X_train_ = self.X_train_.drop([col], axis=1)
        
        col_num = self.X_train_.select_dtypes(exclude=['object'])
        col_obj = self.X_train_.select_dtypes(include=['object'])
        
        for col in col_num:
            norm = np.array(sorted(self.X_train_[col][self.X_train_[col].notnull()]))
            Q1 = np.percentile(norm, 25)
            Q3 = np.percentile(norm, 75)
            IQR = 1.5 * (Q3 - Q1)
            norm = np.array(filter(lambda x: x >= (Q1 - IQR) and x <= (Q3 + IQR), norm))
            
            self.X_means_[col] = np.mean(norm)
            self.X_train_[col] = self.X_train_[col].fillna(self.X_means_[col])
            
            # if self.X_train_[col].nunique() <= 10:
            #     X_dummies = pd.get_dummies(self.X_train_[col], prefix=col)
            #     self.X_train_ = self.X_train_.drop([col], axis=1)
            #     self.X_train_ = pd.concat([self.X_train_, X_dummies], axis=1)
                
            #     self.X_trans_[col] += list(X_dummies.columns)
            
            # else:
                # self.X_train_[col] = zscore(self.X_train_[col], ddof=0)
                
            self.X_trans_[col] += [col]
        
        for col in col_obj:
            self.X_modes_[col] = self.X_train_[col].mode()
            self.X_train_[col] = self.X_train_[col].fillna(self.X_modes_[col])
            
            if self.X_train_[col].nunique() <= 10:
                X_dummies = pd.get_dummies(self.X_train_[col], prefix=col)
                self.X_train_ = self.X_train_.drop([col], axis=1)
                self.X_train_ = pd.concat([self.X_train_, X_dummies], axis=1)
                
                self.X_trans_[col] += list(X_dummies.columns)
            
            else:
                self.X_train_ = self.X_train_.drop([col], axis=1)
        
        print (self.X_train_.shape)
#        print (self.X_train_.head())
        self.PrincipalCA()
        print (self.X_train_.shape)
        print (self.X_train_)
#        pprint(self.X_trans_)
        # pprint(list(self.X_train_.columns))
    
    def cleanse_post(self):
        # pprint(list(self.X_predict_.columns))
        row_count = self.X_predict_.shape[0]
        column_all = list(self.X_predict_.columns)
        
        for key, val in self.X_trans_.iteritems():
            if val == []:
                print 'deleting %s' % key
                self.X_predict_ = self.X_predict_.drop([key], axis=1)
        
        col_num = self.X_predict_.select_dtypes(exclude=['object'])
        col_obj = self.X_predict_.select_dtypes(include=['object'])
        
        for col in col_num:
            self.X_predict_[col] = self.X_predict_[col].fillna(self.X_means_[col])
            
            # if len(self.X_trans_[col]) > 1:
            #     X_dummies = pd.get_dummies(self.X_predict_[col], prefix=col)
            #     for trans in self.X_trans_[col]:
            #         if trans in list(X_dummies.columns):
            #             self.X_predict_ = pd.concat([self.X_predict_, X_dummies[trans]], axis=1)
                    
            #         else:
            #             filler = pd.DataFrame(np.zeros((row_count), dtype=np.int), columns=[trans])
            #             self.X_predict_ = pd.concat([self.X_predict_, filler], axis=1)
            print 'after %s, %d' % (col, self.X_predict_.shape[0])
        
        for col in col_obj:
            self.X_predict_[col] = self.X_predict_[col].fillna(self.X_modes_[col])
            
            if len(self.X_trans_[col]) > 1:
                X_dummies = pd.get_dummies(self.X_predict_[col], prefix=col)
                for trans in self.X_trans_[col]:
                    if trans in list(X_dummies.columns):
                        self.X_predict_ = pd.concat([self.X_predict_, X_dummies[trans]], axis=1)
                    
                    else:
                        filler = pd.DataFrame(np.zeros((row_count), dtype=np.int), columns=[trans])
                        self.X_predict_ = pd.concat([self.X_predict_, filler], axis=1)
                        print 'filler %d' % filler.shape[0]
            
            self.X_predict_ = self.X_predict_.drop([col], axis=1)
            print 'after %s, %d' % (col, self.X_predict_.shape[0])
        
        pprint(list(self.X_predict_.columns))

#def main():
train = pd.read_csv('train.csv')

X = train.drop(['PassengerId', 'Survived'], axis=1)
y = train['Survived']

benchmark = LogisticRegression(random_state=2)

X = X.fillna(X.mean())
X_dummies = pd.get_dummies(X[['Sex', 'Embarked']])
X = X.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
X = pd.concat([X, X_dummies], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

bench_score = cross_val_score(benchmark, X_train, y_train, scoring='accuracy', cv=7)
print 'bench mean-std = %f' % (bench_score.mean() - bench_score.std())

benchmark.fit(X_train, y_train)
print 'bench score = %f' % benchmark.score(X_test, y_test)

X = train.drop(['PassengerId', 'Survived'], axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

aml = AutoML(X_train, y_train)
aml.add_default_models()
aml.cleanse_pref()
aml.search_best_params()
aml.sort_models()
print aml.name_score_

aml.set_params('LogReg', {'random_state': 1})
# print aml.get_params('LogReg')

aml_score = cross_val_score(aml.models_[aml.name_sorted_[0]], aml.X_train_, aml.y_train_, scoring='accuracy', cv=7)
print 'aml mean-std = %f' % (aml_score.mean() - aml_score.std())

aml.fit()
aml.set_predict(X_test, y_test)
aml.cleanse_post()
#print 'aml score = %f' % aml.score()

#if __name__ == '__main__':
#    main()
