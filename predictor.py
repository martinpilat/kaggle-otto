import pandas as pd
import numpy as np
from sklearn import ensemble, calibration, metrics, cross_validation
from sklearn import feature_extraction, preprocessing
import xgboost as xgb
import keras.models as kermod
import keras.layers.core as kerlay
import keras.layers.advanced_activations as keradv
import keras.layers.normalization as kernorm
import scipy.optimize as scopt

# read all the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# encode the labels
le = preprocessing.LabelEncoder()
train_y = train_data.target
train_y = le.fit_transform(train_y)

# drop id and target from the attributes
train_X = train_data.drop('id', axis=1)
train_X = train_X.drop('target', axis=1)

# drop id from the test set
test_X = test_data.drop('id', axis=1)

# simple sklearn compatible wrapper around a keras NN
class NNPredictor:

    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y):

        train_X = self.scaler.fit_transform(X.values.astype(float))
        try:
            train_X = train_X.toarray()
        except:
            pass
        train_y = preprocessing.OneHotEncoder(sparse=False, n_values=9).fit_transform(list(map(lambda x: [x], y)))

        self.nn = kermod.Sequential()

        self.nn.add(kerlay.Dropout(0.1))
        self.nn.add(kerlay.Dense(train_X.shape[1], 1024, init='glorot_uniform'))
        self.nn.add(keradv.PReLU(1024,))
        self.nn.add(kernorm.BatchNormalization((1024,), mode=1))
        self.nn.add(kerlay.Dropout(0.5))

        self.nn.add(kerlay.Dense(1024, 512, init='glorot_uniform'))
        self.nn.add(keradv.PReLU(512,))
        self.nn.add(kernorm.BatchNormalization((512,), mode=1))
        self.nn.add(kerlay.Dropout(0.5))

        self.nn.add(kerlay.Dense(512, 256, init='glorot_uniform'))
        self.nn.add(keradv.PReLU(256,))
        self.nn.add(kernorm.BatchNormalization((256,), mode=1))
        self.nn.add(kerlay.Dropout(0.5))

        self.nn.add(kerlay.Dense(256, 9, init='glorot_uniform', activation='softmax'))
        self.nn.compile(loss='categorical_crossentropy', optimizer='adam')

        # shuffle the training set
        sh = np.array(range(len(train_X)))
        np.random.shuffle(sh)
        train_X = train_X[sh]
        train_y = train_y[sh]

        self.nn.fit(train_X, train_y, nb_epoch=60, batch_size=2048, verbose=0)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        scaled_X = self.scaler.transform(X.values.astype(float))
        try:
            scaled_X = scaled_X.toarray()
        except:
            pass
        return self.nn.predict_proba(scaled_X)

    def get_params(self, deep=False):
        return {}

    def score(self, X, y):
        return metrics.log_loss(y, self.predict_proba(X))


# NN ensemble, 10 NNs
# 5 of them with StandarScaler, 5 of them with TfidfTransformer
class NNEnsemble:

    def fit(self, X, y):

        self.clfs = []
        i = 0
        skf = cross_validation.StratifiedKFold(y, n_folds=10)
        for train_idx, test_idx in skf:
            i += 1
            if i % 2 == 0:
                clf = NNPredictor(scaler=preprocessing.StandardScaler())
            else:
                clf = NNPredictor(scaler=feature_extraction.text.TfidfTransformer())
            clf.fit(X.iloc[train_idx], y[train_idx])
            self.clfs.append(clf)
            print(clf.score(X.iloc[test_idx], y[test_idx]))

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pred = 0
        for clf in self.clfs:
            pred += 0.1*preprocessing.normalize(clf.predict_proba(X), axis=1, norm='l1')

        return pred

    def get_params(self, deep=False):
        return {}

    def score(self, X, y):
        return metrics.log_loss(y, self.predict_proba(X))


# the main predictor, ensemble of NNPredictor, calibrated Random Forest and
# and XGBoost
class OttoPredictor:

    def fit(self, X, y):
        # keep 5% for calibration later
        sss = cross_validation.StratifiedShuffleSplit(y, test_size=0.05)
        for tr, cal in sss:
            break

        # define the two classifiers
        self.clf1 = xgb.XGBClassifier(objective="multi:softprob",
                                      n_estimators=400,
                                      max_depth=8)
        self.clf2 = calibration.CalibratedClassifierCV(
                                    ensemble.RandomForestClassifier(
                                        n_estimators=1000,
                                        n_jobs=8,
                                        class_weight='auto'),
                                    method='isotonic')
        self.clf3 = NNEnsemble()

        # fit the classifiers
        self.clf1.fit(X.iloc[tr], y[tr])
        self.clf2.fit(X.iloc[tr], y[tr])
        self.clf3.fit(X.iloc[tr], y[tr])

        # predict everything before ensembling
        self.pr1 = self.clf1.predict_proba(X.iloc[cal])
        self.pr2 = self.clf2.predict_proba(X.iloc[cal])
        self.pr3 = self.clf3.predict_proba(X.iloc[cal])

        self.pr1 = preprocessing.normalize(self.pr1, axis=1, norm='l1')
        self.pr2 = preprocessing.normalize(self.pr2, axis=1, norm='l1')
        self.pr3 = preprocessing.normalize(self.pr3, axis=1, norm='l1')

        print("XGB log loss:", metrics.log_loss(y[cal], self.pr1))
        print("RF log loss:", metrics.log_loss(y[cal], self.pr2))
        print("NN log loss:", metrics.log_loss(y[cal], self.pr3))
        print("XGB+RF+NN log loss:", metrics.log_loss(y[cal], (self.pr1+self.pr2+self.pr3)/3))

        self.clfs = [self.clf1, self.clf2, self.clf3]

        predictions = []
        for clf in self.clfs:
            predictions.append(clf.predict_proba(X.iloc[cal]))

        self.cal_y = y[cal]

        def log_loss_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = 0
            for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction

            return metrics.log_loss(self.cal_y, final_prediction)

        scores = []
        wghts = []
        for i in range(20):
            if not i:
                starting_values = [1/3]*len(self.clfs)
            else:
                starting_values = np.random.uniform(size=len(self.clfs))

            cons = ({'type': 'eq', 'fun': lambda w: 1-sum(w)})
            bounds = [(0, 1)]*len(predictions)

            res = scopt.minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

            scores.append(res['fun'])
            wghts.append(res['x'])

        bestSC = np.min(scores)
        bestWght = wghts[np.argmin(scores)]
        self.weights = bestWght

        print('Ensamble Score: {best_score}'.format(best_score=bestSC))
        print('Best Weights: {weights}'.format(weights=bestWght))

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pred = 0
        for weight, clf in zip(self.weights, self.clfs):
            pred += weight*clf.predict_proba(X)

        return pred

    def get_params(self, deep=False):
        return {}

    def score(self, X, y):
        return metrics.log_loss(y, self.predict_proba(X))

# train the main predictor

clf = OttoPredictor()
clf.fit(train_X, train_y)

# predict the test sets by smaller batches to reduce the amount of req. memory
preds = [clf.predict_proba(test_X[10000 * i:10000 * (i + 1)]) for i in range(14)]
preds.append(clf.predict_proba(test_X[140000:]))
preds = np.vstack(preds)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('submission.csv', index_label='id')
