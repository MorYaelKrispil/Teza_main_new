
import sys

if sys.platform == "win32":  # noqa
    print(
        "The pyrfr library (requirement of fanova) can currently not be installed on Windows systems"
    )
    exit()

from tabulate import tabulate
import openml
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# naive grid search implementation
import pandas as pd

data_dict = {'Fasion_mnist': 146825, 'riccardo': 168338, 'robert': 168332, 'guillermo': 168337}
n_estimators = [int(x) for x in np.linspace(start=2, stop=100, num=100)]  # 100
# Number of features to consider at every split
max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                1]  # [(x) for x in np.linspace(start = 0.0, stop = 1, num = 10)]

# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=20, num=10)]
# Method of selecting samples for training each tree
bootstrap = [False, True]
criterion = ['gini', 'entropy']

# Create the random grid
# randomforestclassifier__ --> pprint(pipline.steps) #step names + __ need to be added to the parameter name
grid_param = {'n_estimators': n_estimators,
              'n_jobs': [-1],
              'max_features': max_features,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap,
              'criterion': criterion,
              'random_state': [42]
              }
for k, v in enumerate(data_dict):

    task_id = data_dict[v]
    global_algo = 'DQNAgent'
    global_env_name = v  # 'robert on server'a=RandomForestClassifier()
    chkpt_dir = '/Users/zlililaor/Desktop/MorTeza-main/TEZA/Grid_openML_trials/Grid_trials_' + global_env_name + '.csv'
    print('starting' + global_env_name)
    task = openml.tasks.get_task(task_id)

    # splite to trin test
    X, y = task.get_X_and_y(dataset_format="dataframe")
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    counter = 0
    best_score = 0
    Param_df = pd.DataFrame()
    for a in n_estimators:
        for b in max_features:
            for c in min_samples_split:
                for d in min_samples_leaf:
                    for e in bootstrap:
                        for f in criterion:
                            counter = counter + 1
                            # for each combination of parameters, train an SVC
                            rf = RandomForestClassifier(n_estimators=a, max_features=b, min_samples_split=c,
                                                        min_samples_leaf=d, bootstrap=e, criterion=f, random_state=42,
                                                        n_jobs=-1)
                            rf.fit(X_train, y_train)
                            # evaluate the SVC on the test set
                            score = rf.score(X_test, y_test)
                            df = pd.DataFrame(rf.get_params(), index=[0])
                            df['score'] = score

                            # if we got a better score, store the score and parameters
                            if score > best_score:
                                best_score = score
                                best_parameters = {'n_estimators': a, 'max_features': b, 'min_samples_split': c,
                                                   'min_samples_leaf': d, 'bootstrap': e, 'criterion': f,
                                                   'random_state': 42, 'n_jobs': -1}
                                print('best_parameters', best_parameters)
                                print('best_score', best_score)
                                df['best_parameters'] = str(best_parameters)

                            Param_df = Param_df.append(df, ignore_index=True)

                            if counter % 50 == 0:
                                Param_df.to_csv(chkpt_dir)
                                print(tabulate(Param_df.tail(10), headers='keys', tablefmt='psql'))

    print("Best score: {:.2f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))

