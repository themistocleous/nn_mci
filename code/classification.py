#%% Neural Network Application with Cross-Validation
# * Name: Charalambos Themistocleous
# * 2018
#%% Clean Memory before rerunning
for name in dir():
    if not name.startswith('_'): del globals()[name]

#%% libraries for dataset preparation, feature engineering, model training 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition, ensemble
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve, classification_report
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from scipy import interp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy, textblob, string, os
import pandas as pd
SEED = 2000
np.random.seed(SEED)
pd.options.display.max_columns = None

#%% Functions
def evaluate_from_history(history_model):
    """Charalambos Themistocleous 2018"""
    k =1
    accuracy = []
    sd_accuracy = []
    loss = []
    valid_accuracy = []
    sd_valid_accuracy = []
    valid_loss = []

    for i in history_model:
        model_hist = i
        print("Epoch: {}\nAccuracy: {}, Loss: {}, \nValidation Accuracy: {} Validation Loss: {}".format(k,np.mean(model_hist.history['acc']),
                                                                                                        np.mean(model_hist.history['loss']),
                                                                                                        np.mean(model_hist.history['val_acc']),
                                                                                                        np.mean(model_hist.history['val_loss'])))
        accuracy.append(np.mean(model_hist.history['acc']))
        sd_accuracy.append(np.std(model_hist.history['acc']))
        loss.append(np.mean(model_hist.history['loss']))
        valid_accuracy.append(np.mean(model_hist.history['val_acc']))
        valid_accuracy.append(np.mean(model_hist.history['val_acc']))
        sd_valid_accuracy.append(np.std(model_hist.history['val_acc']))
        valid_loss.append(np.mean(model_hist.history['val_loss']))
        k = k+1
    print("======================================================================================")    
    print("\nTotal Results:    \nFinal Accuracy: {}    \nSD Accuracy: {}    \nFinal Loss: {}    \nFinal Validation Accuracy: {}    \nSD Validation Accuracy: {}    \nFinal Validation Loss: {}".format(np.mean(accuracy),
                                        np.std(accuracy),
                                        np.mean(loss),
                                        np.mean(valid_accuracy),
                                        np.std(valid_accuracy),
                                        np.mean(valid_loss)))


#%% :Plots
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = 10, 10

# Where to save the figures
PROJECT_ROOT_DIR = "."
TITLE_ID = "speech_features"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "figures", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

#%%
# Import data
data = pd.read_csv("./data/phone_processed.csv")

#%%
columns =  ['condition', 'speaker','gender',
            'segment','age','duration', 'f0_mean', 'f0_min', 'f0_max',
            'F1.25', 'F1.50', 'F1.75', 'F2.25', 'F2.50', 
            'F2.75', 'F3.25', 'F3.50', 'F3.75', 'F4.25', 
            'F4.50', 'F4.75', 'F5.25', 'F5.50', 'F5.75', 'F1', 'F2', 'F3']

data = data[columns]

# Remove information about consonants
data = data[data["segment"]=="V"]
data = data.drop(["segment"], axis=1)


#%%
# Check for NAs
data.isnull().sum()
data = data.dropna(axis=0, how="any")

# Remap factor labels using numbers
# The change should apply to Condition, Gender, but not Speaker
categories = {"condition" : {'HC' : 0., 'MCI' : 1.},
              "gender" :  {'M' : 0., 'F' : 1.},
              "speaker" : {'MXF3-B9HZ-XPTT': 0 ,'VYGF-G4MF-27S5': 1,'F_84BU-QCK7-2N8M': 2,
                          '7AGX-DJD3-UK39': 3,'STGY-LEC5-AC2H': 4,'JXQ5-T9DF-F75B': 5,
                          '8ZMZ-SBTA-MQRP': 6,'9EL8-FFJJ-TG5R': 7,'V56M-J6RZ-MCLU': 8,
                          'WDWD-XU5H-2EXS': 9,'VDS6-5ULJ-X7YZ': 10,'PMZM-UYEH-ZSZW': 11,
                          'XPEU-R5UC-JYUJ': 12,'F_D59B-5ZYP-SZPQ': 13,'7AL7-ACFY-T3JC': 14,
                          'ZD3N-YVLS-83FK': 15,'S47G-BTHK-MMHG': 16,'R4SL-RNDM-UJCG': 17,
                          'F_79PA-NFUF-3HEQ': 18,'RMHT-ZRLX-S59E': 19,'4Y2B-KLTS-6UWA': 20,
                          'F_NU94-Z6MG-K85T': 21,'ZDTD-KWQQ-JGPB': 22,'YZBC-5BN2-ZWCA': 23,
                          '8LZM-NGHA-PRWM': 24,'TXLH-R239-CPQG': 25,'F_CRG2-7X84-BLWF': 26,
                          'H2D9-Q75Y-3GDT': 27,'WFBN-8TLU-ZKA4': 28,'F_X53K-6TFA-C2ST': 29,
                          'W4N3-62QL-KMZN': 30,'F_FTPL-RAE7-HJM7': 31,'P8X7-9J36-36PQ': 32,
                          'F_FAGL-HXK3-LBSD': 33,'UGLP-E76L-2F98': 34,'GA8A-2E3B-LX6F': 35,
                          'PWPP-D5EX-EJDN': 36,'BPYV-H5DR-66TJ': 37,'F_M9S8-UYGU-Q865': 38,
                          'SFBG-TTUN-B8Y7': 39,'XC2P-PRBL-8REY': 40,'F_A2A7-6E6H-H5GJ': 41,
                          'T4UG-98R9-RDCS': 42,'ZFBE-82GF-SERT': 43,'LABM-75GG-5K4H': 44,
                          'XTQN-WA4V-39Y7': 45,'VCR4-AKGC-MJBK': 46,'RGH4-HSN4-GCMR': 47,
                          'T6NM-JN4H-4PFM': 48,'TAHE-J5KC-XXTL': 49,'M4NC-ELN3-WJBA': 50,
                          'U89K-H28U-NKZ9': 51,'F_PWCG-RX8N-W8NB': 52,'F_8GBU-GFRE-QUZR': 53,
                          'GQ28-X6XM-YSX': 54}
             }

data.replace(categories, inplace=True)


#%%
columns =  ['condition', 'speaker','gender',
            'age','duration', 'f0_mean', 'f0_min', 'f0_max',
            'F1.25', 'F1.50', 'F1.75', 'F2.25', 'F2.50', 
            'F2.75', 'F3.25', 'F3.50', 'F3.75', 'F4.25', 
            'F4.50', 'F4.75', 'F5.25', 'F5.50', 'F5.75','F1','F2','F3']
import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
dfim=imp.fit_transform(data)
data = pd.DataFrame(dfim, columns=columns)
speaker = data.speaker
d = data

d.F1 = np.log(d['F1.50'])
d.F2 = np.log(d['F2.50'])
d.F3 = np.log(d['F3.50'])
d = d.drop(["speaker"], axis=1)
X, y = d.iloc[:,1:].values, d.iloc[:,0].values


#%%
print("Model 1: 1 Hiden Layer")
# 1 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel1_tprs = []
cmodel1_aucs = []
cmodel1_resultsA = []
cmodel1_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model1_cvscores = []
model1_history_main = [] # This will save the results from all cross-validations
model1 = Sequential()
model1.add(Dense(300, input_dim=24, activation='relu'))
model1.add(Dense(300, activation='relu'))   
model1.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model1_history = model1.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel1_y_pred = model1.predict_classes(X_test_transformed)    
    model1_scores = model1.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model1.metrics_names[1], model1_scores[1]*100))
    model1_history_main.append(model1_history)
    
    
    model1_cvscores.append(model1_scores[1] * 100)
    
    # Corrects
    cmodel1_n_correct = sum(cmodel1_y_pred == y_test)
    cmodel1_accuracy2 = model1_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel1_accuracy1 = cmodel1_n_correct / len(cmodel1_y_pred)   
    cmodel1_resultsA.append(cmodel1_accuracy1)
    cmodel1_resultsB.append(cmodel1_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel1_fpr, cmodel1_tpr, cmodel1_thresholds = roc_curve(y_test, cmodel1_y_pred)
    cmodel1_tprs.append(interp(mean_fpr, cmodel1_fpr, cmodel1_tpr))
    cmodel1_tprs[-1][0] = 0.0
    cmodel1_roc_auc = auc(cmodel1_fpr, cmodel1_tpr)
    cmodel1_aucs.append(cmodel1_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel1_resultsA))
    print("sNN Accuracy B: {}".format(cmodel1_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel1_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel1_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel1_resultsA)), np.std(flatten(cmodel1_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel1_resultsB), np.std(cmodel1_resultsB)))


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
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel1_auc, std_cmodel1_auc),
         lw=2.5, alpha=.8)

std_cmodel1_tpr = np.std(cmodel1_tprs, axis=0)
cmodel1_tprs_upper = np.minimum(mean_cmodel1_tpr + std_cmodel1_tpr, 1)
cmodel1_tprs_lower = np.maximum(mean_cmodel1_tpr - std_cmodel1_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel1_tprs_lower, cmodel1_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
model1_cvscores


#%%
history_model1 = model1_history_main
evaluate_from_history(history_model1)


#%%
for i in model1_history_main:
    print(i)


#%%
print("Model 2: 2 Hidden Layers")
# 2 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel2_tprs = []
cmodel2_aucs = []
cmodel2_resultsA = []
cmodel2_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model2_cvscores = []
model2_history_main = [] # This will save the results from all cross-validations
model2 = Sequential()
model2.add(Dense(300, input_dim=24, activation='relu'))
model2.add(Dense(300, activation='relu'))
model2.add(Dense(300, activation='relu'))   
model2.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model2_history = model2.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel2_y_pred = model2.predict_classes(X_test_transformed)    
    model2_scores = model2.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model2.metrics_names[1], model2_scores[1]*100))
    model2_history_main.append(model2_history)
    
    
    model2_cvscores.append(model2_scores[1] * 100)
    
    # Corrects
    cmodel2_n_correct = sum(cmodel2_y_pred == y_test)
    cmodel2_accuracy2 = model2_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel2_accuracy1 = cmodel2_n_correct / len(cmodel2_y_pred)   
    cmodel2_resultsA.append(cmodel2_accuracy1)
    cmodel2_resultsB.append(cmodel2_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel2_fpr, cmodel2_tpr, cmodel2_thresholds = roc_curve(y_test, cmodel2_y_pred)
    cmodel2_tprs.append(interp(mean_fpr, cmodel2_fpr, cmodel2_tpr))
    cmodel2_tprs[-1][0] = 0.0
    cmodel2_roc_auc = auc(cmodel2_fpr, cmodel2_tpr)
    cmodel2_aucs.append(cmodel2_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel2_resultsA))
    print("sNN Accuracy B: {}".format(cmodel2_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel2_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel2_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel2_resultsA)), np.std(flatten(cmodel2_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel2_resultsB), np.std(cmodel2_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel2_resultsA))
print("SNN Accuracy B: {}".format(cmodel2_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel2_tpr = np.mean(cmodel2_tprs, axis=0)
mean_cmodel2_tpr[-1] = 1.0
mean_cmodel2_auc = auc(mean_fpr, mean_cmodel2_tpr)
print("Mean AUC {}".format(mean_cmodel2_auc))
std_cmodel2_auc = np.std(cmodel2_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel2_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel2_auc, std_cmodel2_auc),
         lw=2.5, alpha=.8)

std_cmodel2_tpr = np.std(cmodel2_tprs, axis=0)
cmodel2_tprs_upper = np.minimum(mean_cmodel2_tpr + std_cmodel2_tpr, 1)
cmodel2_tprs_lower = np.maximum(mean_cmodel2_tpr - std_cmodel2_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel2_tprs_lower, cmodel2_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model2_cvscores), np.std(model2_cvscores)))


#%%
history_model2 = model2_history_main
evaluate_from_history(history_model2)


#%%
print("Model 3: 3 Hidden Layers")
# 3 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel3_tprs = []
cmodel3_aucs = []
cmodel3_resultsA = []
cmodel3_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model3_cvscores = []
model3_history_main = [] # This will save the results from all cross-validations
model3 = Sequential()
model3.add(Dense(300, input_dim=24, activation='relu'))
model3.add(Dense(300, activation='relu'))
model3.add(Dense(300, activation='relu'))
model3.add(Dense(300, activation='relu'))   
model3.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model3_history = model3.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel3_y_pred = model3.predict_classes(X_test_transformed)    
    model3_scores = model3.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model3.metrics_names[1], model3_scores[1]*100))
    model3_history_main.append(model3_history)
    
    
    model3_cvscores.append(model3_scores[1] * 100)
    
    # Corrects
    cmodel3_n_correct = sum(cmodel3_y_pred == y_test)
    cmodel3_accuracy2 = model3_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel3_accuracy1 = cmodel3_n_correct / len(cmodel3_y_pred)   
    cmodel3_resultsA.append(cmodel3_accuracy1)
    cmodel3_resultsB.append(cmodel3_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel3_fpr, cmodel3_tpr, cmodel3_thresholds = roc_curve(y_test, cmodel3_y_pred)
    cmodel3_tprs.append(interp(mean_fpr, cmodel3_fpr, cmodel3_tpr))
    cmodel3_tprs[-1][0] = 0.0
    cmodel3_roc_auc = auc(cmodel3_fpr, cmodel3_tpr)
    cmodel3_aucs.append(cmodel3_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel3_resultsA))
    print("sNN Accuracy B: {}".format(cmodel3_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel3_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel3_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel3_resultsA)), np.std(flatten(cmodel3_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel3_resultsB), np.std(cmodel3_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel3_resultsA))
print("SNN Accuracy B: {}".format(cmodel3_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel3_tpr = np.mean(cmodel3_tprs, axis=0)
mean_cmodel3_tpr[-1] = 1.0
mean_cmodel3_auc = auc(mean_fpr, mean_cmodel3_tpr)
print("Mean AUC {}".format(mean_cmodel3_auc))
std_cmodel3_auc = np.std(cmodel3_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel3_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel3_auc, std_cmodel3_auc),
         lw=2.5, alpha=.8)

std_cmodel3_tpr = np.std(cmodel3_tprs, axis=0)
cmodel3_tprs_upper = np.minimum(mean_cmodel3_tpr + std_cmodel3_tpr, 1)
cmodel3_tprs_lower = np.maximum(mean_cmodel3_tpr - std_cmodel3_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel3_tprs_lower, cmodel3_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model3_cvscores), np.std(model3_cvscores)))


#%%
history_model3 = model3_history_main
evaluate_from_history(history_model3)


#%%
print("Model 4: 4 Hidden Layers")
# 4 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel4_tprs = []
cmodel4_aucs = []
cmodel4_resultsA = []
cmodel4_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model4_cvscores = []
model4_history_main = [] # This will save the results from all cross-validations
model4 = Sequential()
model4.add(Dense(300, input_dim=24, activation='relu'))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(300, activation='relu'))
model4.add(Dense(300, activation='relu')) 
model4.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model4.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model4_history = model4.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel4_y_pred = model4.predict_classes(X_test_transformed)    
    model4_scores = model4.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model4.metrics_names[1], model4_scores[1]*100))
    model4_history_main.append(model4_history)
    
    
    model4_cvscores.append(model4_scores[1] * 100)
    
    # Corrects
    cmodel4_n_correct = sum(cmodel4_y_pred == y_test)
    cmodel4_accuracy2 = model4_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel4_accuracy1 = cmodel4_n_correct / len(cmodel4_y_pred)   
    cmodel4_resultsA.append(cmodel4_accuracy1)
    cmodel4_resultsB.append(cmodel4_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel4_fpr, cmodel4_tpr, cmodel4_thresholds = roc_curve(y_test, cmodel4_y_pred)
    cmodel4_tprs.append(interp(mean_fpr, cmodel4_fpr, cmodel4_tpr))
    cmodel4_tprs[-1][0] = 0.0
    cmodel4_roc_auc = auc(cmodel4_fpr, cmodel4_tpr)
    cmodel4_aucs.append(cmodel4_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel4_resultsA))
    print("sNN Accuracy B: {}".format(cmodel4_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel4_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel4_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel4_resultsA)), np.std(flatten(cmodel4_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel4_resultsB), np.std(cmodel4_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel4_resultsA))
print("SNN Accuracy B: {}".format(cmodel4_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel4_tpr = np.mean(cmodel4_tprs, axis=0)
mean_cmodel4_tpr[-1] = 1.0
mean_cmodel4_auc = auc(mean_fpr, mean_cmodel4_tpr)
print("Mean AUC {}".format(mean_cmodel4_auc))
std_cmodel4_auc = np.std(cmodel4_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel4_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel4_auc, std_cmodel4_auc),
         lw=2.5, alpha=.8)

std_cmodel4_tpr = np.std(cmodel4_tprs, axis=0)
cmodel4_tprs_upper = np.minimum(mean_cmodel4_tpr + std_cmodel4_tpr, 1)
cmodel4_tprs_lower = np.maximum(mean_cmodel4_tpr - std_cmodel4_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel4_tprs_lower, cmodel4_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model4_cvscores), np.std(model4_cvscores)))


#%%
history_model4 = model4_history_main
evaluate_from_history(history_model4)


#%%
print("Model 5: 5 Hidden Layers")
# 5 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel5_tprs = []
cmodel5_aucs = []
cmodel5_resultsA = []
cmodel5_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model5_cvscores = []
model5_history_main = [] # This will save the results from all cross-validations
model5 = Sequential()
model5.add(Dense(300, input_dim=24, activation='relu'))
model5.add(Dense(300, activation='relu'))
model5.add(Dense(300, activation='relu'))
model5.add(Dense(300, activation='relu'))
model5.add(Dense(300, activation='relu'))
model5.add(Dense(300, activation='relu'))
model5.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model5.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model5_history = model5.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel5_y_pred = model5.predict_classes(X_test_transformed)    
    model5_scores = model5.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model5.metrics_names[1], model5_scores[1]*100))
    model5_history_main.append(model5_history)
    
    
    model5_cvscores.append(model5_scores[1] * 100)
    
    # Corrects
    cmodel5_n_correct = sum(cmodel5_y_pred == y_test)
    cmodel5_accuracy2 = model5_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel5_accuracy1 = cmodel5_n_correct / len(cmodel5_y_pred)   
    cmodel5_resultsA.append(cmodel5_accuracy1)
    cmodel5_resultsB.append(cmodel5_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel5_fpr, cmodel5_tpr, cmodel5_thresholds = roc_curve(y_test, cmodel5_y_pred)
    cmodel5_tprs.append(interp(mean_fpr, cmodel5_fpr, cmodel5_tpr))
    cmodel5_tprs[-1][0] = 0.0
    cmodel5_roc_auc = auc(cmodel5_fpr, cmodel5_tpr)
    cmodel5_aucs.append(cmodel5_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel5_resultsA))
    print("sNN Accuracy B: {}".format(cmodel5_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel5_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel5_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel5_resultsA)), np.std(flatten(cmodel5_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel5_resultsB), np.std(cmodel5_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel5_resultsA))
print("SNN Accuracy B: {}".format(cmodel5_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel5_tpr = np.mean(cmodel5_tprs, axis=0)
mean_cmodel5_tpr[-1] = 1.0
mean_cmodel5_auc = auc(mean_fpr, mean_cmodel5_tpr)
print("Mean AUC {}".format(mean_cmodel5_auc))
std_cmodel5_auc = np.std(cmodel5_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel5_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel5_auc, std_cmodel5_auc),
         lw=2.5, alpha=.8)

std_cmodel5_tpr = np.std(cmodel5_tprs, axis=0)
cmodel5_tprs_upper = np.minimum(mean_cmodel5_tpr + std_cmodel5_tpr, 1)
cmodel5_tprs_lower = np.maximum(mean_cmodel5_tpr - std_cmodel5_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel5_tprs_lower, cmodel5_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model5_cvscores), np.std(model5_cvscores)))


#%%
history_model5 = model5_history_main
evaluate_from_history(history_model5)


#%%
print("Model 6: 6 Hidden Layers")
# 5 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel6_tprs = []
cmodel6_aucs = []
cmodel6_resultsA = []
cmodel6_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model6_cvscores = []
model6_history_main = [] # This will save the results from all cross-validations
model6 = Sequential()
model6.add(Dense(300, input_dim=24, activation='relu'))
model6.add(Dense(300, activation='relu'))
model6.add(Dense(300, activation='relu'))
model6.add(Dense(300, activation='relu'))
model6.add(Dense(300, activation='relu'))
model6.add(Dense(300, activation='relu'))
model6.add(Dense(300, activation='relu'))
model6.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model6.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model6_history = model6.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel6_y_pred = model6.predict_classes(X_test_transformed)    
    model6_scores = model6.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model6.metrics_names[1], model6_scores[1]*100))
    model6_history_main.append(model6_history)
    
    
    model6_cvscores.append(model6_scores[1] * 100)
    
    # Corrects
    cmodel6_n_correct = sum(cmodel6_y_pred == y_test)
    cmodel6_accuracy2 = model6_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel6_accuracy1 = cmodel6_n_correct / len(cmodel6_y_pred)   
    cmodel6_resultsA.append(cmodel6_accuracy1)
    cmodel6_resultsB.append(cmodel6_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel6_fpr, cmodel6_tpr, cmodel6_thresholds = roc_curve(y_test, cmodel6_y_pred)
    cmodel6_tprs.append(interp(mean_fpr, cmodel6_fpr, cmodel6_tpr))
    cmodel6_tprs[-1][0] = 0.0
    cmodel6_roc_auc = auc(cmodel6_fpr, cmodel6_tpr)
    cmodel6_aucs.append(cmodel6_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel6_resultsA))
    print("sNN Accuracy B: {}".format(cmodel6_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel6_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel6_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel6_resultsA)), np.std(flatten(cmodel6_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel6_resultsB), np.std(cmodel6_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel6_resultsA))
print("SNN Accuracy B: {}".format(cmodel6_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel6_tpr = np.mean(cmodel6_tprs, axis=0)
mean_cmodel6_tpr[-1] = 1.0
mean_cmodel6_auc = auc(mean_fpr, mean_cmodel6_tpr)
print("Mean AUC {}".format(mean_cmodel6_auc))
std_cmodel6_auc = np.std(cmodel6_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel6_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel6_auc, std_cmodel6_auc),
         lw=2.5, alpha=.8)

std_cmodel6_tpr = np.std(cmodel6_tprs, axis=0)
cmodel6_tprs_upper = np.minimum(mean_cmodel6_tpr + std_cmodel6_tpr, 1)
cmodel6_tprs_lower = np.maximum(mean_cmodel6_tpr - std_cmodel6_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel6_tprs_lower, cmodel6_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model6_cvscores), np.std(model6_cvscores)))


#%%
history_model6 = model6_history_main
evaluate_from_history(history_model6)


#%%
print("Model 7: 7 Hidden Layers")
# 7 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel7_tprs = []
cmodel7_aucs = []
cmodel7_resultsA = []
cmodel7_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model7_cvscores = []
model7_history_main = [] # This will save the results from all cross-validations
model7 = Sequential()
model7.add(Dense(300, input_dim=24, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(300, activation='relu'))
model7.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model7.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model7_history = model7.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel7_y_pred = model7.predict_classes(X_test_transformed)    
    model7_scores = model7.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model7.metrics_names[1], model7_scores[1]*100))
    model7_history_main.append(model7_history)
    
    
    model7_cvscores.append(model7_scores[1] * 100)
    
    # Corrects
    cmodel7_n_correct = sum(cmodel7_y_pred == y_test)
    cmodel7_accuracy2 = model7_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel7_accuracy1 = cmodel7_n_correct / len(cmodel7_y_pred)   
    cmodel7_resultsA.append(cmodel7_accuracy1)
    cmodel7_resultsB.append(cmodel7_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel7_fpr, cmodel7_tpr, cmodel7_thresholds = roc_curve(y_test, cmodel7_y_pred)
    cmodel7_tprs.append(interp(mean_fpr, cmodel7_fpr, cmodel7_tpr))
    cmodel7_tprs[-1][0] = 0.0
    cmodel7_roc_auc = auc(cmodel7_fpr, cmodel7_tpr)
    cmodel7_aucs.append(cmodel7_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel7_resultsA))
    print("sNN Accuracy B: {}".format(cmodel7_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel7_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel7_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel7_resultsA)), np.std(flatten(cmodel7_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel7_resultsB), np.std(cmodel7_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel7_resultsA))
print("SNN Accuracy B: {}".format(cmodel7_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel7_tpr = np.mean(cmodel7_tprs, axis=0)
mean_cmodel7_tpr[-1] = 1.0
mean_cmodel7_auc = auc(mean_fpr, mean_cmodel7_tpr)
print("Mean AUC {}".format(mean_cmodel7_auc))
std_cmodel7_auc = np.std(cmodel7_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel7_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel7_auc, std_cmodel7_auc),
         lw=2.5, alpha=.8)

std_cmodel7_tpr = np.std(cmodel7_tprs, axis=0)
cmodel7_tprs_upper = np.minimum(mean_cmodel7_tpr + std_cmodel7_tpr, 1)
cmodel7_tprs_lower = np.maximum(mean_cmodel7_tpr - std_cmodel7_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel7_tprs_lower, cmodel7_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model7_cvscores), np.std(model7_cvscores)))


#%%
history_model7 = model7_history_main
evaluate_from_history(history_model7)


#%%
print("Model 8: 8 Hidden Layers")
# 8 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel8_tprs = []
cmodel8_aucs = []
cmodel8_resultsA = []
cmodel8_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model8_cvscores = []
model8_history_main = [] # This will save the results from all cross-validations
model8 = Sequential()
model8.add(Dense(300, input_dim=24, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(300, activation='relu'))
model8.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model8.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model8_history = model8.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel8_y_pred = model8.predict_classes(X_test_transformed)    
    model8_scores = model8.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model8.metrics_names[1], model8_scores[1]*100))
    model8_history_main.append(model8_history)
    
    
    model8_cvscores.append(model8_scores[1] * 100)
    
    # Corrects
    cmodel8_n_correct = sum(cmodel8_y_pred == y_test)
    cmodel8_accuracy2 = model8_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel8_accuracy1 = cmodel8_n_correct / len(cmodel8_y_pred)   
    cmodel8_resultsA.append(cmodel8_accuracy1)
    cmodel8_resultsB.append(cmodel8_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel8_fpr, cmodel8_tpr, cmodel8_thresholds = roc_curve(y_test, cmodel8_y_pred)
    cmodel8_tprs.append(interp(mean_fpr, cmodel8_fpr, cmodel8_tpr))
    cmodel8_tprs[-1][0] = 0.0
    cmodel8_roc_auc = auc(cmodel8_fpr, cmodel8_tpr)
    cmodel8_aucs.append(cmodel8_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel8_resultsA))
    print("sNN Accuracy B: {}".format(cmodel8_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel8_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel8_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel8_resultsA)), np.std(flatten(cmodel8_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel8_resultsB), np.std(cmodel8_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel8_resultsA))
print("SNN Accuracy B: {}".format(cmodel8_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel8_tpr = np.mean(cmodel8_tprs, axis=0)
mean_cmodel8_tpr[-1] = 1.0
mean_cmodel8_auc = auc(mean_fpr, mean_cmodel8_tpr)
print("Mean AUC {}".format(mean_cmodel8_auc))
std_cmodel8_auc = np.std(cmodel8_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel8_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel8_auc, std_cmodel8_auc),
         lw=2.5, alpha=.8)

std_cmodel8_tpr = np.std(cmodel8_tprs, axis=0)
cmodel8_tprs_upper = np.minimum(mean_cmodel8_tpr + std_cmodel8_tpr, 1)
cmodel8_tprs_lower = np.maximum(mean_cmodel8_tpr - std_cmodel8_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel8_tprs_lower, cmodel8_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model8_cvscores), np.std(model8_cvscores)))


#%%
history_model8 = model8_history_main
evaluate_from_history(history_model8)


#%%
print("Model 9: 9 Hidden Layers")
# 5 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel9_tprs = []
cmodel9_aucs = []
cmodel9_resultsA = []
cmodel9_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model9_cvscores = []
model9_history_main = [] # This will save the results from all cross-validations
model9 = Sequential()
model9.add(Dense(300, input_dim=24, activation='relu'))
#1
model9.add(Dense(300, activation='relu'))
#2
model9.add(Dense(300, activation='relu'))
#3
model9.add(Dense(300, activation='relu'))
#4
model9.add(Dense(300, activation='relu'))
#5
model9.add(Dense(300, activation='relu'))
#6
model9.add(Dense(300, activation='relu'))
#7
model9.add(Dense(300, activation='relu'))
#8
model9.add(Dense(300, activation='relu'))
#9
model9.add(Dense(300, activation='relu'))
model9.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model9.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model9_history = model9.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel9_y_pred = model9.predict_classes(X_test_transformed)    
    model9_scores = model9.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model9.metrics_names[1], model9_scores[1]*100))
    model9_history_main.append(model9_history)
    
    
    model9_cvscores.append(model9_scores[1] * 100)
    
    # Corrects
    cmodel9_n_correct = sum(cmodel9_y_pred == y_test)
    cmodel9_accuracy2 = model9_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel9_accuracy1 = cmodel9_n_correct / len(cmodel9_y_pred)   
    cmodel9_resultsA.append(cmodel9_accuracy1)
    cmodel9_resultsB.append(cmodel9_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel9_fpr, cmodel9_tpr, cmodel9_thresholds = roc_curve(y_test, cmodel9_y_pred)
    cmodel9_tprs.append(interp(mean_fpr, cmodel9_fpr, cmodel9_tpr))
    cmodel9_tprs[-1][0] = 0.0
    cmodel9_roc_auc = auc(cmodel9_fpr, cmodel9_tpr)
    cmodel9_aucs.append(cmodel9_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel9_resultsA))
    print("sNN Accuracy B: {}".format(cmodel9_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel9_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel9_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel9_resultsA)), np.std(flatten(cmodel9_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel9_resultsB), np.std(cmodel9_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel9_resultsA))
print("SNN Accuracy B: {}".format(cmodel9_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel9_tpr = np.mean(cmodel9_tprs, axis=0)
mean_cmodel9_tpr[-1] = 1.0
mean_cmodel9_auc = auc(mean_fpr, mean_cmodel9_tpr)
print("Mean AUC {}".format(mean_cmodel9_auc))
std_cmodel9_auc = np.std(cmodel9_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel9_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel9_auc, std_cmodel9_auc),
         lw=2.5, alpha=.8)

std_cmodel9_tpr = np.std(cmodel9_tprs, axis=0)
cmodel9_tprs_upper = np.minimum(mean_cmodel9_tpr + std_cmodel9_tpr, 1)
cmodel9_tprs_lower = np.maximum(mean_cmodel9_tpr - std_cmodel9_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel9_tprs_lower, cmodel9_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model9_cvscores), np.std(model9_cvscores)))


#%%
history_model9 = model9_history_main
evaluate_from_history(history_model9)


#%%
print("Model 10: 10 Hidden Layers")
# 5 Layers
from sklearn.model_selection import GroupKFold
flatten = lambda mylist: [item for sublist in mylist for item in sublist]
cmodel10_tprs = []
cmodel10_aucs = []
cmodel10_resultsA = []
cmodel10_resultsB = []
mean_fpr = np.linspace(0, 1, 100)

group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, speaker)
print(group_kfold,) 
model10_cvscores = []
model10_history_main = [] # This will save the results from all cross-validations
model10 = Sequential()
model10.add(Dense(300, input_dim=24, activation='relu'))
#1
model10.add(Dense(300, activation='relu'))
#2
model10.add(Dense(300, activation='relu'))
#3
model10.add(Dense(300, activation='relu'))
#4
model10.add(Dense(300, activation='relu'))
#5
model10.add(Dense(300, activation='relu'))
#6
model10.add(Dense(300, activation='relu'))
#7
model10.add(Dense(300, activation='relu'))
#8
model10.add(Dense(300, activation='relu'))
#9
model10.add(Dense(300, activation='relu'))
#10
model10.add(Dense(300, activation='relu'))
model10.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model10.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

for train_index, test_index in group_kfold.split(np.log(X), y, speaker):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    # Transform X_Test  
    X_test_transformed = scaler.transform(X_test)

    model10_history = model10.fit(X_train_transformed, y_train, validation_data=(X_test_transformed, y_test),epochs=80, batch_size=35)


    # Evaluate classifiers
    cmodel10_y_pred = model10.predict_classes(X_test_transformed)    
    model10_scores = model10.evaluate(X_test_transformed, y_test, verbose=1)
    print("%s: %.2f%%" % (model10.metrics_names[1], model10_scores[1]*100))
    model10_history_main.append(model10_history)
    
    
    model10_cvscores.append(model10_scores[1] * 100)
    
    # Corrects
    cmodel10_n_correct = sum(cmodel10_y_pred == y_test)
    cmodel10_accuracy2 = model10_scores[1]  # 1 is validation accuracy, 0 is accuracy
    #
    cmodel10_accuracy1 = cmodel10_n_correct / len(cmodel10_y_pred)   
    cmodel10_resultsA.append(cmodel10_accuracy1)
    cmodel10_resultsB.append(cmodel10_accuracy2)

    
    print("==========================================================================")
    print("FOLD {}".format(i))
    print("==========================================================================")
        # Compute ROC curve and area the curve RF
    cmodel10_fpr, cmodel10_tpr, cmodel10_thresholds = roc_curve(y_test, cmodel10_y_pred)
    cmodel10_tprs.append(interp(mean_fpr, cmodel10_fpr, cmodel10_tpr))
    cmodel10_tprs[-1][0] = 0.0
    cmodel10_roc_auc = auc(cmodel10_fpr, cmodel10_tpr)
    cmodel10_aucs.append(cmodel10_roc_auc)


    #i += 1
    print("sNN Accuracy A: {}".format(cmodel10_resultsA))
    print("sNN Accuracy B: {}".format(cmodel10_resultsB))
    print("sNN ROC_AUC: {}".format(cmodel10_roc_auc))
    print("sNN Confusion Matrix: \n{}".format(confusion_matrix(y_test, cmodel10_y_pred)))
    
print("==========================================================================")
print("FINAL RESULTS")
print("==========================================================================") 
print("SNN Mean {}, SD {}".format(np.mean(flatten(cmodel10_resultsA)), np.std(flatten(cmodel10_resultsA))))
print("SNN Mean {}, SD {}".format(np.mean(cmodel10_resultsB), np.std(cmodel10_resultsB)))


print("==========================================================================")    
print("Accuracy summaries")
print("SNN Accuracy A: {}".format(cmodel10_resultsA))
print("SNN Accuracy B: {}".format(cmodel10_resultsB))
print("Acurracy A is manually calculated")
print("==========================================================================")    

# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)

mean_cmodel10_tpr = np.mean(cmodel10_tprs, axis=0)
mean_cmodel10_tpr[-1] = 1.0
mean_cmodel10_auc = auc(mean_fpr, mean_cmodel10_tpr)
print("Mean AUC {}".format(mean_cmodel10_auc))
std_cmodel10_auc = np.std(cmodel10_aucs)


# PLOT the mean
plt.plot(mean_fpr, mean_cmodel10_tpr, color='r',
         label=r'NN ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel10_auc, std_cmodel10_auc),
         lw=2.5, alpha=.8)

std_cmodel10_tpr = np.std(cmodel10_tprs, axis=0)
cmodel10_tprs_upper = np.minimum(mean_cmodel10_tpr + std_cmodel10_tpr, 1)
cmodel10_tprs_lower = np.maximum(mean_cmodel10_tpr - std_cmodel10_tpr, 0)

# PLOT
plt.fill_between(mean_fpr, cmodel10_tprs_lower, cmodel10_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightgray")
save_fig("Model_",1)
plt.show()


#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(model10_cvscores), np.std(model10_cvscores)))


#%%
history_model10 = model10_history_main
evaluate_from_history(history_model10)


#%%
# PLOT THE BASELINE
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray',
         label='Baseline', alpha=.8)



# PLOT the mean
plt.plot(mean_fpr, mean_cmodel1_tpr, color='r',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel1_auc, std_cmodel1_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel2_tpr, color='b',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel2_auc, std_cmodel2_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel3_tpr, color='g',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel3_auc, std_cmodel3_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel4_tpr, color='m',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel4_auc, std_cmodel4_auc),
         lw=2.5, alpha=.8)


plt.plot(mean_fpr, mean_cmodel5_tpr, color='#0066cc',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel5_auc, std_cmodel5_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel6_tpr, color='#cc0099',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel6_auc, std_cmodel6_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel7_tpr, color='#ff9900',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel7_auc, std_cmodel7_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel8_tpr, color='#8000ff',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel8_auc, std_cmodel8_auc),
         lw=2.5, alpha=.8)

plt.plot(mean_fpr, mean_cmodel9_tpr, color='#00bfff',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel9_auc, std_cmodel9_auc),
         lw=2.5, alpha=.8)


plt.plot(mean_fpr, mean_cmodel10_tpr, color='#4d2e00',
         label=r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_cmodel10_auc, std_cmodel10_auc),
         lw=2.5, alpha=.8)



# PLOT
#plt.fill_between(mean_fpr, cmodel1_tprs_lower, cmodel1_tprs_upper, color='r', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel2_tprs_lower, cmodel2_tprs_upper, color='b', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel3_tprs_lower, cmodel3_tprs_upper, color='g', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel4_tprs_lower, cmodel4_tprs_upper, color='m', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel5_tprs_lower, cmodel5_tprs_upper, color='#0066cc', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel6_tprs_lower, cmodel6_tprs_upper, color='#cc0099', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel7_tprs_lower, cmodel7_tprs_upper, color='#ff9900', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
plt.fill_between(mean_fpr, cmodel8_tprs_lower, cmodel8_tprs_upper, color='#8000ff', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel9_tprs_lower, cmodel9_tprs_upper, color='#00bfff', alpha=.2)#, label=r'$\pm$ 1 std. dev.')
#plt.fill_between(mean_fpr, cmodel10_tprs_lower, cmodel10_tprs_upper, color='#4d2e00', alpha=.2)#, label=r'$\pm$ 1 std. dev.')





plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('svPPA')
plt.legend(loc="lower right")
plt.grid(color="lightblue")
save_fig("Model_",1)
plt.show()


#%%
print("\n\n\n======\nModel 1\n======\n")
evaluate_from_history(history_model1)
print("\n\n\n======\nModel 2\n======\n")
evaluate_from_history(history_model2)
print("\n\n\n======\nModel 3\n======\n")
evaluate_from_history(history_model3)
print("\n\n\n======\nModel 4\n======\n")
evaluate_from_history(history_model4)
print("\n\n\n======\nModel 5\n======\n")
evaluate_from_history(history_model5)
print("\n\n\n======\nModel 6\n======\n")
evaluate_from_history(history_model6)
print("\n\n\n======\nModel 7\n======\n")
evaluate_from_history(history_model7)
print("\n\n\n======\nModel 8\n======\n")
evaluate_from_history(history_model8)
print("\n\n\n======\nModel 9\n======\n")
evaluate_from_history(history_model9)
print("\n\n\n======\nModel 10\n======\n")
evaluate_from_history(history_model10)


#%%



