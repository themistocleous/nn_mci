'''
Neural Network Application with Cross-Validation
Name: Charalambos Themistocleous
Year: 2018
Example of the Basic Architecture of the Network. Multible NeuralNets with different number of layers are compared

'''
# %% libraries for dataset preparation, feature engineering, model training
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition, ensemble
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve, classification_report
from sklearn.impute import SimpleImputer
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

SEED = 2000
np.random.seed(SEED)
pd.options.display.max_columns = None

# Plotting Parameters
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = 10, 10

FIGURE_DIR = "../figures/"
TITLE_ID = "NN"


def save_fig(figure_id, TITLE_ID, FIGURE_DIR, tight_layout=True):
    '''Save figures figures with the number of the NN architecture applied'''
    import os
    if figure_id == "":
        print("No Figure ID was provided.")
    else:
        pass

    if TITLE_ID == "":
        TITLE_ID = "Figure"
    else:
        TITLE_ID = TITLE_ID

    if FIGURE_DIR == "":
        FIGURE_DIR = "../figures"
    else:
        FIGURE_DIR = FIGURE_DIR

    try:
        os.makedirs(FIGURE_DIR)
    except FileExistsError:
        # directory already exists
        pass

    path = os.path.join(FIGURE_DIR, TITLE_ID + "_" + figure_id + ".png")
    print("Saving figure No", figure_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# Import data
data = pd.read_csv("./data/phone_processed.csv")

# Define Columns
columns = ['condition', 'speaker', 'gender',
           'segment', 'age', 'duration', 'f0_mean', 'f0_min', 'f0_max',
           'F1.25', 'F1.50', 'F1.75', 'F2.25', 'F2.50',
           'F2.75', 'F3.25', 'F3.50', 'F3.75', 'F4.25',
           'F4.50', 'F4.75', 'F5.25', 'F5.50', 'F5.75', 'F1', 'F2', 'F3']
data = data[columns]

# Remove information about consonants
data = data[data["segment"] == "V"]
data = data.drop(["segment"], axis=1)

# Check for NAs
data.isnull().sum()
data = data.dropna(axis=0, how="any")

# Remap factor labels using numbers
categories = {"condition": {'HC': 0., 'MCI': 1.},
              "gender":  {'M': 0., 'F': 1.},
              "speaker": {'MXF3-B9HZ-XPTT': 0, 'VYGF-G4MF-27S5': 1, 'F_84BU-QCK7-2N8M': 2,
                          '7AGX-DJD3-UK39': 3, 'STGY-LEC5-AC2H': 4, 'JXQ5-T9DF-F75B': 5,
                          '8ZMZ-SBTA-MQRP': 6, '9EL8-FFJJ-TG5R': 7, 'V56M-J6RZ-MCLU': 8,
                          'WDWD-XU5H-2EXS': 9, 'VDS6-5ULJ-X7YZ': 10, 'PMZM-UYEH-ZSZW': 11,
                          'XPEU-R5UC-JYUJ': 12, 'F_D59B-5ZYP-SZPQ': 13, '7AL7-ACFY-T3JC': 14,
                          'ZD3N-YVLS-83FK': 15, 'S47G-BTHK-MMHG': 16, 'R4SL-RNDM-UJCG': 17,
                          'F_79PA-NFUF-3HEQ': 18, 'RMHT-ZRLX-S59E': 19, '4Y2B-KLTS-6UWA': 20,
                          'F_NU94-Z6MG-K85T': 21, 'ZDTD-KWQQ-JGPB': 22, 'YZBC-5BN2-ZWCA': 23,
                          '8LZM-NGHA-PRWM': 24, 'TXLH-R239-CPQG': 25, 'F_CRG2-7X84-BLWF': 26,
                          'H2D9-Q75Y-3GDT': 27, 'WFBN-8TLU-ZKA4': 28, 'F_X53K-6TFA-C2ST': 29,
                          'W4N3-62QL-KMZN': 30, 'F_FTPL-RAE7-HJM7': 31, 'P8X7-9J36-36PQ': 32,
                          'F_FAGL-HXK3-LBSD': 33, 'UGLP-E76L-2F98': 34, 'GA8A-2E3B-LX6F': 35,
                          'PWPP-D5EX-EJDN': 36, 'BPYV-H5DR-66TJ': 37, 'F_M9S8-UYGU-Q865': 38,
                          'SFBG-TTUN-B8Y7': 39, 'XC2P-PRBL-8REY': 40, 'F_A2A7-6E6H-H5GJ': 41,
                          'T4UG-98R9-RDCS': 42, 'ZFBE-82GF-SERT': 43, 'LABM-75GG-5K4H': 44,
                          'XTQN-WA4V-39Y7': 45, 'VCR4-AKGC-MJBK': 46, 'RGH4-HSN4-GCMR': 47,
                          'T6NM-JN4H-4PFM': 48, 'TAHE-J5KC-XXTL': 49, 'M4NC-ELN3-WJBA': 50,
                          'U89K-H28U-NKZ9': 51, 'F_PWCG-RX8N-W8NB': 52, 'F_8GBU-GFRE-QUZR': 53,
                          'GQ28-X6XM-YSX': 54}
              }
data.replace(categories, inplace=True)

# Imputing
imp = SimpleImputer(missing_values=np.nan, strategy='median')
dfim = imp.fit_transform(data)
columns = ['condition', 'speaker', 'gender',
           'age', 'duration', 'f0_mean', 'f0_min', 'f0_max',
           'F1.25', 'F1.50', 'F1.75', 'F2.25', 'F2.50',
           'F2.75', 'F3.25', 'F3.50', 'F3.75', 'F4.25',
           'F4.50', 'F4.75', 'F5.25', 'F5.50', 'F5.75', 'F1', 'F2', 'F3']
data = pd.DataFrame(dfim, columns=columns)
speaker = data.speaker

d = data
d.F1 = np.log(d['F1.50'])
d.F2 = np.log(d['F2.50'])
d.F3 = np.log(d['F3.50'])
d = d.drop(["speaker"], axis=1)
X, y = d.iloc[:, 1:].values, d.iloc[:, 0].values
# 1 Layer


def flatten(mylist): return [item for sublist in mylist for item in sublist]


cmodel1_tprs = []
cmodel1_aucs = []
cmodel1_resultsA = []
cmodel1_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,)
model1_cvscores = []
model1_history_main = []  # This will save the results from all cross-validations

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# MODEL 1
MODEL_NO = "1"


model1 = Sequential()
model1.add(Dense(300, input_dim=24, activation='relu'))
model1.add(Dense(300, activation='relu'))
model1.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

i = 0  # Folds Counter

for train_index, test_index in group_kfold.split(X, y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test
    X_test_transformed = scaler.transform(X_test)

    model1_history = model1.fit(X_train_transformed, y_train, validation_data=(
        X_test_transformed, y_test), epochs=80, batch_size=35)

    # Evaluate classifiers
    cmodel1_y_pred = model1.predict_classes(X_test_transformed)
    model1_scores = model1.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model1.metrics_names[1], model1_scores[1]*100))
    model1_history_main.append(model1_history)

    model1_cvscores.append(model1_scores[1] * 100)

    # Corrects
    cmodel1_n_correct = sum(cmodel1_y_pred == y_test)
    # 1 is validation accuracy, 0 is accuracy
    cmodel1_accuracy2 = model1_scores[1]
    #
    cmodel1_accuracy1 = cmodel1_n_correct / len(cmodel1_y_pred)
    cmodel1_resultsA.append(cmodel1_accuracy1)
    cmodel1_resultsB.append(cmodel1_accuracy2)

    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
    # Compute ROC curve and area the curve RF
    cmodel1_fpr, cmodel1_tpr, cmodel1_thresholds = roc_curve(
        y_test, cmodel1_y_pred)
    cmodel1_tprs.append(interp(mean_fpr, cmodel1_fpr, cmodel1_tpr))
    cmodel1_tprs[-1][0] = 0.0
    cmodel1_roc_auc = auc(cmodel1_fpr, cmodel1_tpr)
    cmodel1_aucs.append(cmodel1_roc_auc)

    i += 1
    print("sNN Accuracy A: {}".format(cmodel1_resultsA))
    print("sNN Accuracy B: {}".format(cmodel1_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel1_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(
        confusion_matrix(y_test, cmodel1_y_pred)))

print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================")
print("SNN Mean {}, SD {}".format(
    np.mean(flatten(cmodel1_resultsA)), np.std(flatten(cmodel1_resultsA))))
print("SNN Mean {}, SD {}".format(
    np.mean(cmodel1_resultsB), np.std(cmodel1_resultsB)))


print("==========================================================================")
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel1_resultsA))
print("SNN Accuracy B: {}".format(cmodel1_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel1_tpr = np.mean(cmodel1_tprs, axis=0)
mean_cmodel1_tpr[-1] = 1.0
mean_cmodel1_auc = auc(mean_fpr, mean_cmodel1_tpr)
print("Mean AUC {}".format(mean_cmodel1_auc))
std_cmodel1_auc = np.std(cmodel1_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel1_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (
             mean_cmodel1_auc, std_cmodel1_auc),
         lw=2.5, alpha=.8)

std_cmodel1_tpr = np.std(cmodel1_tprs, axis=0)
cmodel1_tprs_upper = np.minimum(mean_cmodel1_tpr + std_cmodel1_tpr, 1)
cmodel1_tprs_lower = np.maximum(mean_cmodel1_tpr - std_cmodel1_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel1_tprs_lower, cmodel1_tprs_upper,
                 color='r', alpha=.2)  # , label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MCI/Healthy Individuals')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig(TITLE_ID=TITLE_ID, FIGURE_DIR=FIGURE_DIR, figure_id=MODEL_NO)
plt.show()
