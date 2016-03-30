## Drop a column in dataframe
df.drop('loan_yn', axis=1, inplace=True)

## Assign 1 and 0 to binary variable with a new column
df['loan_yn'] = 0
df.loc[df.loan == 'yes', 'loan_yn'] = 1

## Table
df.groupby(['loan_status', 'target']).loan_status.count().groupby(level=['loan_status','target']).value_counts()

#display the breakout
df.groupby(['grade'])['target'].mean()
df.groupby(['sub_grade']).target.count()
df.groupby(['prestige']).target.value_counts(sort=False)
# crosstab prestige 1 admission
# frequency table cutting prestige and whether or not someone was admitted
pd.crosstab(handCalc['d_prestige_1.0'], df.admit, rownames=['Prestige 1'], colnames=['admit'], margins=True)


# Create dummy variable
dummy_grade = pd.get_dummies(df['grade'], prefix='d_grade')
df_lr = df_lr.join(dummy_grade.ix[:, :'d_grade_F'])

#Normalized/Standardize Matrix:
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X)
X_scaled                                          



# Splitting Training and Testing Sets
from sklearn.cross_validation import train_test_split
## Split the Original/Unbalanced Data using train_test_split function:
X = df.values
y = df.y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=2016)
X.shape
y.shape


## Cross Validation
## Verifying result using Cross Validation on the entire Data Set: n_fold=5
from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold(y, n_folds=5, random_state=2016)

true_cv = []
pred_cv = []
accu_cv = []
for train_index, test_index in skf:
    y_pred = logistic_regression (X_num[train_index], X_num[test_index], y[train_index], y[test_index], zero_weight)
    true_cv.append(y[test_index])
    pred_cv.append(y_pred)
    accu_cv.append(accuracy_score(y[test_index], y_pred))

## Here we plot out the confusion matrix of the Cross-Validation Results
## However, note that the importance of the result is the Overall Increased ROI printed above.    
TrueLabel = list(itertools.chain(*true_cv))
PredictedLabel = list(itertools.chain(*pred_cv))
print ('Correlation between the actual and prediction is:', pearsonr(TrueLabel, PredictedLabel)[0], \
       'with p-value',  ("%2.2f" % pearsonr(TrueLabel, PredictedLabel)[1]))

cm = confusion_matrix(PredictedLabel, TrueLabel)
fig, ax = plt.subplots()
im = ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.title('Confusion matrix')
fig.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Scatter Plot using Heat Map to Show Correlation
import seaborn as sns
sns.set(style="white")
corr = df[['loan_amnt','int_rate','sub_grade','annual_inc','open_acc',\
           'dti','fico_range_high','revol_bal','revol_util','Avg_Median',\
           'Pop']].corr()  # Compute the correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)               # Generate a mask for the upper triangle
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))                     # Set up the matplotlib figure
cmap = sns.diverging_palette(220, 10, as_cmap=True)       # Generate a custom diverging colormap
sns.heatmap(corr)                      # Draw the heatmap with the mask and correct aspect ratio
plt.show()




#logistic regression (Customized)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
y_pre = logistic_regression (os_X_num_train, X_num_test, os_y_train, y_test)
def logistic_regression (X_train, X_test, y_train, y_test, zero_weight=1):
    """Perform logistic regression using Sklearn package
    Note, must already import LogisticRegression from sklearn.linear_model package
    Arguments:
    X_train -- The predictor-only array dataset for training the model
    X_test  -- The predictor-only array dataset for testing the trained model
    y_train -- The response-only array for training the model
    y_test  -- The response-only array for testing the model results
    zero_weight -- The weight to used for regression's class_weight parameter
                   for favoring predicint y=0; use positive integer only
                   Default value is 1 (i.e. no weight)
    """
    ## Fit the logistic regression model with class_wieght 1X-10X on y=0 (favoring y=0)    
    lr = LogisticRegression(class_weight={0: zero_weight})
    lr.fit(X_train, y_train)
    ## Predict test set target values using weighted model and compared accuracy
    y_predicted = lr.predict(X_test)
    confusion = pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True)
    a, b = confusion.shape
    a -= 1
    print(confusion)
    print('The MODELED accuracy score of the test/valdiation set is {:2.3}%'.format(accuracy_score(y_test, y_predicted)*100))
    print('The MODELED accuracy on predicted good loans of test/valid. set is {:2.3f}% with {:2.3f}% reduced coverage'.format(confusion[1][a-1]/confusion[1][a]*100, (1-confusion[1][a]/confusion['All'][a])*100))
    print('The ACTUAL accuracy score of the test/validation set is {:2.3}%'.format(np.count_nonzero(y_test==1)/len(y_test)*100))
    roi_num_test_pred = X_test[:,1] * y_predicted * y_test
    print('The PREDICTED Annualized ROI of test/validation set on predicted good loans is: {:2.3f}%'.format(roi_num_test_pred.mean()))
    roi_num_test = X_test[:,1] * y_test
    print('The ACTUAL Annualized ROI of test/validation set on overall true good loans is: {:2.3f}%'.format(roi_num_test.mean()))
    print('\n')    
    return (y_predicted)

	
	

##3D Scatter Plot