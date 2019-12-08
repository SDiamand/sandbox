%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, make_scorer, mean_squared_error, jaccard_score
import prince
import statistics
import operator
import pickle

def plot_evolution(df_payments, suptitle, filename, periodtype, color):
    if periodtype is 'terminate_date':
        periods = max(df_payments[periodtype]).month + 12
    elif periodtype is 'register_date':
        periods = max(df_payments[periodtype]).month

    df_count = pd.DataFrame()
    df_count['period'] = pd.date_range('2016-01', periods=periods, freq='M', closed='right')
    df_count['period'] =  pd.to_datetime(df_count['period']).dt.to_period('M')

    df_count['count_all'] = 0
    df_count['count_bad'] = 0
    df_count['sum_paid'] = 0
    df_count['sum_non_paid'] = 0

    for index, row in df_count.iterrows():
        count = 0
        count_bad = 0
        sum_paid = 0
        sum_non_paid = 0
        for index2, row2 in df_payments.iterrows():
            if (row['period'].year ==  row2[periodtype].year) & (row['period'].month ==  row2[periodtype].month):
                count += 1
                sum_paid += row2['paid_funds']
                sum_non_paid += row2['non_paid_funds']
                if (row['period'].year ==  row2[periodtype].year) & (row['period'].month ==  row2[periodtype].month) & (row2['bad_clients'] == 1):
                    count_bad += 1
        df_count.at[index, 'count_all'] = count
        df_count.at[index, 'count_bad'] = count_bad
        df_count.at[index, 'sum_paid'] = sum_paid
        df_count.at[index, 'sum_non_paid'] = sum_non_paid

    df_count['ratio_bad_clients'] = df_count['count_bad'] / df_count['count_all']
    df_count['ratio_non_paid'] = df_count['sum_non_paid'] / (df_count['sum_non_paid'] + df_count['sum_paid'])

    # plot total bad clients and proportion of bad clients of other types
    fig, ax = plt.subplots(2,2, figsize=(10,12))
    fig.suptitle(suptitle)
    df_count.plot(x='period', y='count_bad', kind='line', legend=False, c=color, title='Count bad clients', ax=ax[0,0])
    df_count.plot(x='period', y='ratio_bad_clients', kind='line', legend=False, c=color, title='Poportion of bad clients', ylim=(0,1), ax=ax[0,1])
    df_count.plot(x='period', y='sum_non_paid', kind='line', legend=False, c=color, title='Count of non-paid funds', ax=ax[1,0])
    df_count.plot(x='period', y='ratio_non_paid', kind='line', legend=False, c=color,title='Poportion of non-paid funds', ylim=(0,1), ax=ax[1,1])
    return fig

raw_df = pd.read_excel('C://Users//Sela//Downloads//Assignment_data.xlsx')

'''
-------------------------------------------------------------------------------
    data exploration and pre-processing
-------------------------------------------------------------------------------
'''

raw_df.shape
print('Available columns: ', raw_df.columns)
print('Variable types: ', raw_df.dtypes)


''' missing values '''
# proportion of missing data out of total number of observations
missing_values = 100 * raw_df.isnull().sum()/raw_df.shape[0]
print('Proportion of features with missing data out of total number of observations (features without missing observations are excluded):\n', missing_values[missing_values>0].sort_values(ascending=False))

df = raw_df
# inspect and consider dropping variables with more than 50% missing observations
    #variable name: warning_pdf_sent
    #variable converted to be represented with alternative dummy variables
df['warning_pdf_sent'].head(30)
df['warning_pdf_sent'].nunique()
df['warning_pdf_sent_dummy_sent'] = df.apply(lambda x: 0 if pd.isnull(x['warning_pdf_sent']) else 1, axis=1)
df = df.drop(columns=['warning_pdf_sent'])
    #variable name: last_penalty_payment
    #variable converted to be represented with alternative dummy variables
df['last_penalty_payment'].head(30)
df['last_penalty_payment'].nunique()
df['last_penalty_payment_dummy_paid'] = df.apply(lambda x: 0 if pd.isnull(x['last_penalty_payment']) else 1, axis=1)
df = df.drop(columns=['last_penalty_payment'])

    #variable name: inkasso_debt
    # fill in 0 for clients without existing debt
df['inkasso_debt'].head(30)
df['inkasso_debt'].nunique()
df['inkasso_debt'] = df['inkasso_debt'].fillna(0)

    #varible name: inkasso_penalty
    # fill in 0 for clients without penalties
df['inkasso_penalty'].head(30)
df['inkasso_penalty'].nunique()
df['inkasso_penalty'] = df['inkasso_penalty'].fillna(0)

    #variable name: Profession_Mode
    # perform one-hot encoding. All zeros would reflect cases that previously had missing data to account for the case that data missing not in random
df['Profession_Mode'].head()
df['Profession_Mode'].nunique()
df = pd.get_dummies(df, columns=['Profession_Mode'], dummy_na=False)

    #variable name: loan_rate_id
    # perform one-hot encoding. All zeros would reflect cases that previously had missing data
df['loan_rate_id'].head()
df['loan_rate_id'].nunique()
df = pd.get_dummies(df, columns=['loan_rate_id'], dummy_na=False)

    #variable name: ip_provider
    # business logic begs one to argue that the IP provider of a client would have little to do with
    # the propensity to predict the client's propensity to pay a loan. This hypothesis is strengethened
    # by the fact that Spain (where clients in the database originate from) is developed country
    # where ICT is accessible.
    # Furthermore, the values in the variable are text strings. Due to constrained computing resources one-hot encoding isn't an option.
    # If it wasn't for the time constraints of the porject I would research techniques to convert the text to ML-friendly numbers with algorithms
    # such as word2vec. For the time being I proceed to drop the variable from the analysis.
    # I would research ways to convert the
df['ip_provider'].head(30)
df['ip_provider'].nunique()
df = df.drop(columns=['ip_provider'])

    #variable name: ip_country
    # One input is quite peculiar: EU # Country is really world wide.
    # Due to the incomprehensiability of the input rows including this input are dropped (1170 values).
    # Rows are to be dropped rather than imputed due to concerns that there is a systematic reason for this input to be extraordinary
    # other inputs are one-encoded
df['ip_country'].head()
df['ip_country'].nunique()
df['ip_country'].unique()
len(df[df['ip_country'] == 'EU # Country is really world wide'])
df = df[df['ip_country'] != 'EU # Country is really world wide']
df = pd.get_dummies(df, columns=['ip_country'], drop_first=True)

    #variable name: verified_by_reg
    # variable is one-hot encoded with 0 values for both 'documents' or 'instantor' refer to NaN whereby the client's documents weren't verified
df['verified_by_reg'].head()
df['verified_by_reg'].nunique()
df = pd.get_dummies(df, columns=['verified_by_reg'], dummy_na=False)

    #varoable name: Province
    # due to the low number of missing values it is assumed that the missing observations are not missing systematically
    # and therefore can be dropped from the sample
df['Province'].head()
df['Province'].nunique()
df['Province'].isnull().sum()
df = df.dropna(subset=['Province'])
df = pd.get_dummies(df, columns=['Province'], drop_first=True)

# verify all missing observations were evaluated
df.isnull().sum()

''' further prepare variables for pre-processing '''
    #variable name: descrStatus
    # convert one-hot encoding
df['descrStatus'].head()
df['descrStatus'].unique()
df['descrStatus_dummy_bad_debtors'] = df.apply(lambda x: 1 if x['descrStatus'] == 'Bad Debtors' else 0, axis=1)

    #variable name: register_date, init_date, terminate_date
    # create new variable - days_from_register_to_init to test whether time between registration to loan disbursement affects or predicts repayment
df['register_date'].head()
len(df[df['init_date']=='0000-00-00 00:00:00'])
df['days_from_register_to_init'] = df[df['descrStatus'] != 'Refused']['init_date'].dt.date - df[df['descrStatus'] != 'Refused']['register_date'].dt.date
df['days_from_register_to_init'] = df['days_from_register_to_init'].fillna(0)
df['days_from_register_to_init'].describe()
register_date = df['register_date']
df = df.drop(columns=['init_date', 'register_date'])

# create new variable - days_from_terminate_to_return
# NaN variables exist for clients that were refused a loan and clients that haven't paid back the loan
df['days_from_terminate_to_return'] = pd.to_datetime(df[(df['descrStatus'] != 'Refused') & (df['descrStatus'] != 'Bad Debtors')]['return_date'], errors='coerce').dt.date - df[(df['descrStatus'] != 'Refused') & (df['descrStatus'] != 'Bad Debtors')]['terminate_date'].dt.date
terminate_date = df['terminate_date']
df = df.drop(columns=['return_date','terminate_date'])

# create new variable - bad_clients
# variable indicates whether the client hasn't paid his/her debt at all or he has paid it but at least 14 days late
# drop days_from_termiante_to_return to avoid perfect collinearity when bad_clients is the target
# proportion of clients who are identified as ''bad'' is 17.7%
df['bad_clients'] = df.apply(lambda x: 1 if ((x['days_from_terminate_to_return'] >= pd.Timedelta(14,'D')) | (x['descrStatus_dummy_bad_debtors']==1)) else 0, axis=1)
df = df.drop(columns=['days_from_terminate_to_return'])
len(df[df['bad_clients'] == 1]) / len(df['bad_clients'])

    # variable s_sms is one level factor and there is dropped
df['s_sms'].nunique()
df = df.drop(columns=['s_sms'])

    #variable name: b_bank_id
    # although accessibility to banking in Spain is high in global terms, bank affiliation nonetheless could capture unobservable socio-economic factors
    # create dummy variables out of b_bank_id to enable processing
df = pd.get_dummies(df, columns=['b_bank_id'], drop_first=True)

    #variable name: b_account
    # drop variable as bank account is distinct to every client
df = df.drop(columns=['b_account'])

    #variable name: Pcode
    # drop variable as DNI/NIE number is likely to be unique to every client
df = df.drop(columns=['Pcode'])

    #variable name: DniOrNie
    # create dummy variable to allow for models that rely on distance measurement for clustering anaylsis
df = pd.get_dummies(df, columns=['DniOrNie'], drop_first=True)

    #variable name: Gender
    # create dummy variable to allow for models that rely on distance measurement for clustering anaylsis
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    #variable name: Da_PostIndex
    # this variable is to be used in the analysis only if the numering is ordinal and not nominal.
    # without information as to the nature of post index the analysis proceeds to drop the variable
df = df.drop(columns=['Da_PostIndex'])

    #variable name: blacklisted
    # drop rows that have been blacklisted and drop the column because it is almost one-level factor
df = df[df['blacklisted'] != 1]
df = df.drop(columns=['blacklisted'])

    #variable verified_by
    # creat dummies to account for the type of verification/application-status. This could hypothetically
    # provide hints as to the quality of the client
df = pd.get_dummies(df, columns=['verified_by'], drop_first=True)

    #variable name: notPaidProlong
    # drop variable because data is one-level factor and has no variance
df['notPaidProlong'].describe()
df = df.drop(columns=['notPaidProlong'])

# final processing before scaling the dataframe
df['days_from_register_to_init'] = pd.to_numeric(df['days_from_register_to_init'])
df = df.drop(columns=['ucid', 'userid', 'descrStatus_dummy_bad_debtors'])

# drop the following variables because although they correlate with the bad_clients they are unlikely
# to indicate a causal realtions (reverse causal if anything)
df = df.drop(columns=['inkasso_penalty', 'inkasso_debt'])

# align feature type with data type
df.loc[:,['allow_by_instantor']]              = df.loc[:,['allow_by_instantor']].astype('uint8')
df.loc[:,['warning_pdf_sent_dummy_sent']]     = df.loc[:,['warning_pdf_sent_dummy_sent']].astype('uint8')
df.loc[:,['last_penalty_payment_dummy_paid']] = df.loc[:,['last_penalty_payment_dummy_paid']].astype('uint8')
df.loc[:,['warning_pdf_sent_dummy_sent']]     = df.loc[:,['warning_pdf_sent_dummy_sent']].astype('uint8')
df.loc[:,['allow_by_instantor']]              = df.loc[:,['allow_by_instantor']].astype('uint8')
df.loc[:,['last_penalty_payment_dummy_paid']] = df.loc[:,['last_penalty_payment_dummy_paid']].astype('uint8')

for col in df.columns:
    if df[col].dtype == 'int64':
        df.loc[:,[col]] = df.loc[:,[col]].astype('float64')

''' scale data so that clustering algorithms relying on distance measurement can be ustalized '''

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['float64'])), index=df.index, columns=df.select_dtypes(include=['float64']).columns)
df_scaled = df_scaled.merge(df.select_dtypes(exclude=['float64']), left_index=True, right_index=True, how='left')
df_scaled = df_scaled.drop(columns=['descrStatus'])

''' visualize realtions between features '''

# plot Pearson correlation heatmap
corr = df_scaled.loc[:,df_scaled.dtypes == 'float64'].corr()
sns.set(style="whitegrid")
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True)).set_title('Pearson correlation heatmap of non-binary variables')
plt.savefig('Pearson_correlation_heatmap.png', bbox_inches='tight')

'''
--------------------------------------------------------------------------------
    phase 1: feature selection: filter method
    variables with low or no variance below a threshold are exluded prior to the analysis
--------------------------------------------------------------------------------
'''

target = 'bad_clients'

X_labels, y_label = df_scaled.columns[df_scaled.columns != target], df_scaled.columns[df_scaled.columns == target]
X, y = df_scaled.loc[:, df_scaled.columns != target], df_scaled.loc[:, df_scaled.columns == target]
y['descrStatus'] = df['descrStatus']
X.shape

# remove binary variables with zero variance in more than var_threshold of obervations
var_threshold = 0.97
feature_selection_1 = VarianceThreshold(threshold=(var_threshold * (1 - var_threshold)))
feature_selection_1.fit(X.select_dtypes(include=['uint8']))
binary_cols_list = X.select_dtypes(include=['uint8'])[X.select_dtypes(include=['uint8']).columns[feature_selection_1.get_support(indices=True)]]

# drop columns with no variance in more than var_threshold of observations
drop_binary_cols_list = list(set(list(X.select_dtypes(include=['uint8']))) - set(binary_cols_list))
X = X.drop(drop_binary_cols_list, axis=1)

X.shape
X.select_dtypes(include=['uint8']).columns.nunique()
X.select_dtypes(include=['float64']).columns.nunique()

'''
--------------------------------------------------------------------------------
    Dimenasionality reduction and visualize segmentation for bad/good clients
    Since data is still high in dimensions. We first reduce dimensionality using
    FAMD because data has continuous as well as categorical data.
--------------------------------------------------------------------------------
'''

# FAMD demands categorical variables to be identified as such
df_famd = X
df_famd['bad_clients'] = y.iloc[:,0]
df_famd['descrStatus'] = y.iloc[:,1]

for col in df_famd.select_dtypes(include=['uint8']).columns:
    df_famd[col] = df_famd[col].astype('category')

famd = prince.FAMD(n_components=5, n_iter=100, copy=True, check_input=False, engine='sklearn', random_state=1)
famd = famd.fit(df_famd.drop(columns=['bad_clients', 'descrStatus'], axis=1))
famd.explained_inertia_

# first graph decpits all descStatus categories for clients while the second graph compares bad clients
#(i.e. those who either never repaid the loan or paid it after more than 14 days) with the rest of the bad_clients
# Client type 1: bad clients
#
#from the FAMD analysis it's can be seen cluters cannot be reduced visually in a convenient way. This means that when
# choosing a model one must consider that clients won't segment as neatly as hoped
fig, ax = plt.subplots(1,2, figsize=(10,8))
famd.plot_row_coordinates(df_famd, x_component=0, y_component=1,
                               color_labels=['Client type {}'.format(t) for t in df_famd['descrStatus']],
                               ellipse_outline=False, ellipse_fill=True,
                               show_points=True, ax=ax[0])

famd.plot_row_coordinates(df_famd, x_component=0, y_component=1,
                               color_labels=['Client type {}'.format(t) for t in df_famd['bad_clients']],
                               ellipse_outline=False, ellipse_fill=True,
                               show_points=True, ax=ax[1])
plt.savefig('FADM_visual_dimensionality_reduction_for_descrStatus_and_bad_clients.png', bbox_inches='tight')

'''
--------------------------------------------------------------------------------
    split data to train-test samples
--------------------------------------------------------------------------------
'''

X = X.drop(columns=['bad_clients','descrStatus'])
X_labels, y_label = X.columns, target
X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.3, random_state=1, stratify=y[target], shuffle=True)

'''
--------------------------------------------------------------------------------
    phase 2: feature selection and learning: wrapper method
    Model AdaBoost using a Random Forest estimator while streamlining hyperparameters
--------------------------------------------------------------------------------
'''

cv = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
max_depths = [5,10]
learning_rates_list = [0.5, 1]
max_features = ['sqrt', 'log2']

f1_scorer = make_scorer(f1_score)
estimator = []
f1 = []

# semi-manual nested grid search cross-validation employing Random Forest as a base estimator for an AdaBoost classifier
# since we don't know whether the financial institution is strategically inclined to minimize type I or type II error the analysis
# chose a neutral position whereby the harmonic mean (F1) score is to be maximized
for max_feature in max_features:
    for max_depth in max_depths:
        rf = RandomForestClassifier(n_estimators=10, max_depth=max_depth, max_features=max_feature, min_samples_split=0.05, n_jobs=-1, random_state=1)
        for learning_rate in learning_rates_list:
            ada = AdaBoostClassifier(base_estimator=rf,
                                    algorithm='SAMME',
                                    random_state=1,
                                    n_estimators=50,
                                    learning_rate=learning_rate)
            ada_cv = cross_validate(ada, X, y[target], cv=cv, n_jobs=-1, scoring=f1_scorer, return_estimator=True)
            f1.append(ada_cv['test_score'])
            estimator.append(ada_cv)

# retrieve model with the highest F1 score
max_f1 = []
for score in f1:
    max_f1.append(statistics.mean(score))

index, value = max(enumerate(max_f1), key=operator.itemgetter(1))

print('Highest F1 score: {}'.format(value))

feature_importances = []
for idx,est in enumerate(estimator[index]['estimator']):
    feature_importances.append(pd.DataFrame(est.feature_importances_,
                                                        index = X_labels,
                                                        columns=['importance']).sort_values('importance', ascending=False))

df_feature_importances = pd.DataFrame((feature_importances[0] + feature_importances[1] + feature_importances[2]),
                                                        index=X_labels,
                                                        columns=['importance']).sort_values('importance', ascending=False)
df_feature_importances = df_feature_importances / 3

# drop features that explain less than 1% of the data
df_feature_importances['importance'] = df_feature_importances.values
df_feature_importances = df_feature_importances.reset_index()
df_feature_importances.to_excel('feature_importance.xlsx')
df_feature_importances = df_feature_importances[df_feature_importances['importance'] >= 0.01]

# plot feature importance of feature explaining more than 1% of observations
fig, ax = plt.subplots()
sns.set_color_codes("pastel")
sns.barplot(x='importance', y='index', data=df_feature_importances, color="b").set_title('Feature Importance')
ax.set_xlabel('')
ax.set_ylabel('')
plt.savefig('feature_importance.png', bbox_inches='tight')

'''
--------------------------------------------------------------------------------
    build predictor model for good/bad clients
    based on the best performing model above

    the model is able to successfully distinguish a bad from good clients
    with an F1 score of 0.90
--------------------------------------------------------------------------------
'''

X_train_model = pd.DataFrame()
X_test_model = pd.DataFrame()

for column in df_feature_importances['index'].values:
    if column in X_train.columns:
        X_train_model[column] = X_train[column]
        X_test_model[column] = X_test[column]

estimator[index]['estimator']
rf = RandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', min_samples_split=0.05, n_jobs=-1, random_state=1)
ada = AdaBoostClassifier(base_estimator=rf, algorithm='SAMME', random_state=1, n_estimators=50, learning_rate=0.5)
ada.fit(X_train_model, y_train)
y_pred = ada.predict(X_test_model)
f1_score(y_test, y_pred)

# save model to disk
pickle.dump(ada, open('predict_bad_clients_model.sav', 'wb'))

'''
--------------------------------------------------------------------------------
    visualize evolution of finance operations through bad clients and non-paid funds
--------------------------------------------------------------------------------
'''

df_payments = pd.DataFrame()
df_payments['non_paid_funds'] = df['notPaidCapital'] + df['notPaidCommission'] + df['notPaidOverpay']
df_payments['paid_funds'] = df['paidCapital'] + df['paidCommission'] + df['paidOverpay']
df_payments['terminate_date'] = pd.to_datetime(terminate_date, errors='coerce').dt.date
df_payments['register_date'] = pd.to_datetime(register_date, errors='coerce').dt.date
df_payments['bad_clients'] = df['bad_clients']
df_payments['descrStatus'] = df['descrStatus']
df_payments = df_payments.sort_values(by='terminate_date')

suptitle = 'Finance operations evolution through bad clients (clients with 14 days overdue pay)\nand\nevolution of non-paid funds (principal, interest and overpay fee)'
filename = 'finance_operations_evolution.png'
periodtype = 'terminate_date'
fig = plot_evolution(df_payments, suptitle, filename, periodtype, 'b')
fig.savefig(filename, bbox_inches='tight')

'''
--------------------------------------------------------------------------------
    Visualize evolution of quality of clients.

    This may help project light to possible seasonailities in the quality of clients
    recruited over time
--------------------------------------------------------------------------------
'''

month_list = pd.DataFrame(pd.to_datetime(df_payments['register_date']).dt.to_period('M').unique())
c = ['r','g','b','y','c','m']
for index, month in month_list.iterrows():
    df_i = df_payments[pd.to_datetime(df_payments['register_date']).dt.to_period('M') == month[0]]
    filename = 'evolution_of_quality_of_clients_enlisted_in_{}.png'.format(month[0])
    suptitle = 'Evolution of quality of client cohort {}: clients with 14 days overdue pay\nand\nnon-paid funds (principal, interest and overpay fee)'.format(month[0])
    fig = plot_evolution(df_i, suptitle, filename, periodtype, c[index])
    fig.savefig(filename, bbox_inches='tight')

'''
-------------------------------------------------------------------------------
    Build a model to predict amount of non-paid funds for a given borrower

    Limitations in the data:
        - Cross-sectional data does allow to utilize longitude observations of client
          behavior that may serve as proxies for unobservables.
        - Data doesn't seem to exhibit ''natural'' segmentation to good/bad clients.
        - Loan/borrowing-related data is likely to violate statistical assumptions
          such as independence and identical distribution of observations and variables.
          Observations are therefore likely to correlate and nullify the viability
          of identifying trends.

    Due to the limitations listed above a trend analysis is avoided until better
    quality data is made available. Instead, an AdaBoost with RF as base estimator model is employed.
    The AdaBoost model is better suited to return predictions for a given client by
    assigning weight to errors returned by subsequent calculations that are commonly used to make predictions on
    data that cannot be explaiend with linear trend analysis.
-------------------------------------------------------------------------------
'''

target_npf = 'non_paid_funds'

# reconsutrct unscaled train-split samples for interpretability
X_npf = pd.DataFrame()
y_npf = pd.DataFrame()
y_npf['bad_clients'] = df['bad_clients']
for column in df_feature_importances['index'].values:
    if column in df.columns:
        X_npf[column] = df[column]

y_npf[target_npf] = X_npf['notPaidCapital'] + X_npf['notPaidCommission']
X_npf = X_npf.drop(columns=['notPaidCapital', 'notPaidCommission'])
X_train_npf, X_test_npf, y_train_npf, y_test_npf = train_test_split(X_npf, y_npf[target_npf], test_size=0.3, random_state=1, shuffle=True)

X_train_npf = X_train_npf.drop(columns=['notPaidCapital', 'notPaidCommission'])
X_test_npf = X_test_npf.drop(columns=['notPaidCapital', 'notPaidCommission'])

max_depths = [5,10]
learning_rates_list = [0.5, 1]
max_features = ['sqrt', 'log2']

estimator = {}
score = []

for max_feature in max_features:
    for max_depth in max_depths:
        rf = RandomForestRegressor(n_estimators=20, max_depth=max_depth, max_features=max_feature, min_samples_split=0.05, n_jobs=-1, random_state=1)
        for learning_rate in learning_rates_list:
            ada_npf = AdaBoostRegressor(base_estimator=rf,
                                    loss='linear',
                                    random_state=1,
                                    n_estimators=50,
                                    learning_rate=learning_rate)
            ada_npf.fit(X_train_npf, y_train_npf)
            estimator[ada_npf] = ada_npf.score(X_test_npf, y_test_npf)

best_ada_npf = max(estimator, key=estimator.get)
print('The model explains {:.2f}% of the variation in notPaidCapital and notPaidCommission.'.format(max(estimator.values())*100))
pickle.dump(best_ada_npf, open('predict_non_paid_funds_model.sav', 'wb'))

'''
-------------------------------------------------------------------------------
    Jaccard Index of bad and refused clients

    Largely spearking, bad clients are clients that de-fact ought have been
    refused a loan. To better classify clients, the analysis proceeds to compare
    bad clients with clients that have been refused a loan. The hypoethesis
    being that these two classes of individuals may have similar qualities
    the financial institution may be interested in focusing identifying. Therefore
    the analysis will proceed to measure the degree of overlap between bad and
    refused clients using the Jaccard index.
-------------------------------------------------------------------------------
'''

df_b = df[df['bad_clients'] == 1]
df_r = df[df['descrStatus'] == 'Refused']

df_b = df_b.drop(columns=['descrStatus', 'bad_clients'])
df_r = df_r.drop(columns=['descrStatus', 'bad_clients'])

df_j_b = pd.DataFrame.copy(df_b)
df_j_r = pd.DataFrame.copy(df_r)

# preparaing bad_clints dataframe for Jaccard coefficient analysis
for column in df_b:
    col_mean = df_b[column].mean()
    for index, row in df_b.iterrows():
            if df_b.loc[index,column] > col_mean:
                df_j_b.at[index, column] = 1
            else:
                df_j_b.at[index, column] = 0

# preparaing refused clients dataframe for Jaccard coefficient analysis
for column in df_r:
    col_mean = df_r[column].mean()
    for index, row in df_r.iterrows():
            if df_r.loc[index,column] > col_mean:
                df_j_r.at[index, column] = 1
            else:
                df_j_r.at[index, column] = 0
# match size of df_j_r to df_j_b
df_j_r = df_j_r.sample(n=df_j_b.shape[0], random_state=1)

df_j = pd.DataFrame(df_j_r.columns.transpose())

for index,row in df_j.iterrows():
    df_j.at[index,'jaccard_index'] = jaccard_score(df_j_r[row], df_j_b[row],average='micro')

print('There are {} variables with more than 99% similtary between bad and refused groups.'.format(df_j[df_j['jaccard_index'] > 0.99].shape[0]))

df_j = df_j.sort_values(by=['jaccard_index'], ascending=False)
df_j.to_excel('jaccard_similarity.xlsx')

fig, ax = plt.subplots(figsize=(15,15))
sns.set_color_codes("pastel")
sns.barplot(x='jaccard_index', y=0, data=df_j[df_j['jaccard_index']>0.99], color="b").set_title('Jaccard similarity (>99%)')
ax.set_xlim(0.9,1)
ax.set_xticks([0.9,0.95,1])
ax.set_xlabel('')
ax.set_ylabel('')
plt.savefig('Jaccard_similarity.png', bbox_inches='tight')
