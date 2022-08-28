import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import pickle

pos_data = np.load('posture_data.npy')
X = pos_data[:, :-1]
y = pos_data[:, -1]
clf = LogisticRegressionCV(cv=5, random_state=0)
clf.fit(X, y)

model_file = 'model.sav'
pickle.dump(clf, open(model_file, 'wb'))
