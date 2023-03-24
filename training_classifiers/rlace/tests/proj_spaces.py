

import pickle as pkl

import numpy as np
from sklearn.linear_model import SGDClassifier
from get_data import get_data
from sklearn.metrics import confusion_matrix
from random import seed

EVAL_CLF_PARAMS = {"loss": "log_loss", "tol": 1e-4, "iters_no_change": 15, "alpha": 1e-4, "max_iter": 25000}
NUM_CLFS_IN_EVAL = 1  # change to 1 for large dataset / high dimensionality


seed(42)

def get_projection(P, rank):
    D, U = np.linalg.eigh(P)
    U = U.T
    W = U[-rank:]
    P_final = np.eye(P.shape[0]) - W.T @ W
    return P_final



#path = "/home/dkletz/tmp/pycharm_project_99/2022-23/Interventions/with_neg_eval/negation_intervention_and_test/training_classifiers/rlace"

path = "/data/dkletz/Experiences/negation_intervention_and_test/training_classifiers"

with open(f"{path}/Proj_50000.pkl", "rb") as f:
    P = pkl.load(f)

p = P['P']
pp = np.eye(p.shape[0], p.shape[1]) - p




def init_classifier():
    return SGDClassifier(loss=EVAL_CLF_PARAMS["loss"], fit_intercept=True, max_iter=EVAL_CLF_PARAMS["max_iter"],
                         tol=EVAL_CLF_PARAMS["tol"], n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
                         n_jobs=32, alpha=EVAL_CLF_PARAMS["alpha"])




X, y = get_data()


svm = init_classifier()

svm.fit(X, y)
score_original = svm.score(X, y)

print(f"Original score: {score_original}")



y_score_orig = svm.predict(X)
a = confusion_matrix(y,  y_score_orig)

print(f"confusion matrix: {a}")

print('\n\n#####\n\n')

svm = init_classifier()
svm.fit(X[:] @ p, y[:])
score_projected_no_svd = svm.score(X @ p, y)
y_score_projected_no_svd = svm.predict(X @ p)


print(f"Projected score : {score_projected_no_svd}")

a = confusion_matrix(y,  y_score_projected_no_svd)

print(f"confusion matrix: {a}")


seed(42)
print('\n\n#####\n\n')
svm = init_classifier()
ot_X = X - X @ p

svm.fit(ot_X, y[:])
score_projected_OT = svm.score(ot_X, y)

y_score_projected_no_svd = svm.predict(X @ p)


print(f"Projected score (ot): {score_projected_OT}")


y_score_orig_ot = svm.predict(ot_X)
a = confusion_matrix(y,  y_score_orig_ot)

print(f"confusion matrix: {a}")


print('\n\n#####\n\n')


seed(42)
svm = init_classifier()
anti_P_X = X @ pp

svm.fit(anti_P_X, y[:])
score_projected_AT = svm.score(anti_P_X, y)


print(f"Projected score (anti P): {score_projected_AT}")



y_score_orig_antiP = svm.predict(anti_P_X)
a = confusion_matrix(y,  y_score_orig_antiP)



print(f"confusion matrix: {a}")



'''


print('\n\n##############################\n##############################\n\n')



print(X.shape)

print(p.shape)


print((X @ p) == np.matmul(X, p))

print((X @ p).shape)


vals = X @ (X - (X @ p)).T

vals = np.diagonal(vals)
print(vals)
print(vals.shape)
signs = np.sign(vals)


svm = init_classifier()

vals = vals.reshape(-1,1)

svm.fit(vals, y[:])
score_projected_val = svm.score(vals, y)
print(f"Projected score (vals): {score_projected_val}")


print(vals[y == 0])
print(vals[y == 1])

print(signs)

signs_corr = signs < 0

c = signs_corr.astype(int)
print(c)

a = confusion_matrix(y,  c)

print(f"confusion matrix: {a}")
'''
