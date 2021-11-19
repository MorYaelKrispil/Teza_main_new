#########################################################################IMPORT##################################################
# %matplotlib inline
import sklearn
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from IPython.display import display
from sklearn.model_selection import RandomizedSearchCV

from torch.utils.tensorboard import SummaryWriter
# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json
from IPython.display import clear_output
from collections import OrderedDict
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import pandas as pd
import pprint
# import ipdb
import timeit
import seaborn as sns
from sklearn import metrics, tree
from sklearn.metrics import accuracy_score, make_scorer, classification_report, confusion_matrix, make_scorer, \
    roc_auc_score, mean_squared_error
import os
import numpy as np
from scipy import optimize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import collections
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, OneCycleLR
import time
import sys
import numpy as np
import collections
import openml
from openml.extensions.sklearn import cat, cont

from collections import OrderedDict
from collections import namedtuple
from itertools import product


#########################################################################Functions##################################################
def data_split(X, y, stratify_by=None):
    global X_train_main
    global X_test_main
    global y_train
    global y_test
    scaler = StandardScaler()
    X_train_main, X_test_main, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,
                                                                  stratify=stratify_by)
    # scalling data 0-1 new 2\01\2021
    # X_train_scaled = scaler.fit_transform(X_train_main)
    # X_test_scaled = scaler.fit_transform(X_test_main)
    # return X_train_scaled, X_test_scaled, y_train, y_test
    return X_train_main, X_test_main, y_train, y_test


# Function that create the episode data - sample randomaly
def get_data(episode_size, policy, mode):
    global dataset
    if mode == 'train':
        dataset = Train.sample(n=episode_size, replace=True)
    else:
        # dataset = train in test -->after getting hyperparameters we will check it on test
        dataset = Train
    return dataset


# Function that separate the episode data into features and label
def data_separate(dataset):
    global X
    global y
    X = dataset.iloc[:, 0:dataset.shape[1] - 1]  # all rows, all the features and no labels
    y = dataset.iloc[:, -1]  # all rows, label only
    return X, y


def Learner(X_train_main, X_test_main, y_train, y_test, learner_model, parameter_dict):
    # global learner
    # global y_pred
    if learner_model == 'DT':
        learner = tree.DecisionTreeClassifier()
    elif learner_model == 'RF_Classifier':
        learner = RandomForestClassifier()
    elif learner_model == 'RF_Regressor':
        learner = RandomForestRegressor()
        parameter_dict['n_jobs'] = -1

    elif learner_model == 'KNN':
        learner = KNeighborsClassifier(metric='hamming', n_neighbors=5)
    elif learner_model == 'SVM':
        learner = SVC()
    elif learner_model == 'NB':
        learner = MultinomialNB()
    elif learner_model == 'AB':
        learner = AdaBoostClassifier()
    elif learner_model == 'GB':
        learner = GradientBoostingClassifier()
    elif learner_model == 'gbdt':
        learner = LGBMRegressor(boosting='gbdt', n_jobs=-1)
    elif learner_model == 'ExtraTreesClassifier':
        learner = ExtraTreesClassifier()
    parameter_dict['random_state'] = 42

    clf = learner.set_params(**parameter_dict)
    # for amazon open ML check
    # create pipline
    cat_imp = make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse=False))
    # create pipline
    cont_imp = SimpleImputer(strategy="most_frequent")
    ct = ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
    pipline = make_pipeline(
        ColumnTransformer([("cat", cat_imp, cat), ("cont", cont_imp, cont)])
        , VarianceThreshold()
        , clf

    )
    pipline.fit(X_train_main, y_train)
    # acc=pipline.score(X_test_main, y_test,cv=2).mean() #added CV +pipline as in openml best score
    acc = cross_val_score(pipline, X_test_main, y_test, cv=2).mean()
    # clf.fit(X_train_main, y_train)
    # acc=clf.score(X_test_main, y_test)
    err = 1 - acc
    print('error: ', err)
    print('-------------------------------------------------------------')
    print('parameter_dict:')
    print('-------------------------------------------------------------')
    pprint.pprint(parameter_dict)
    print('-------------------------------------------------------------')
    return abs(err)


def change_numeric_parameter_scal(dict):
    """
    get x: hyper parameter, in range [a,b]
    retern y: (state), in range ([0, 1]
    how:y = (x-a)/(b-a)

    x = (b-a)y + a

    actions (y-space): +0.1, -0.1,+0.01,+-0.01

        """

    num_actions = 0
    # init_dict={}
    # action_map={}
    # cat_bool_dict={}
    # numeric_param_norm_init_dict={}
    # min_max_numeric_field_dict={}

    init_dict = OrderedDict()
    action_map = OrderedDict()
    cat_bool_dict = OrderedDict()
    numeric_param_norm_init_dict = OrderedDict()
    min_max_numeric_field_dict = OrderedDict()

    # calc num actions
    for k, v in enumerate(dict):
        if type(dict[v]['init_val']) not in [str, bool]:  # if numeric filed actions space is: +-0.1,+-0.01
            num_actions = num_actions + 4
        else:
            num_actions = num_actions + len(dict[v][
                                                'range_val'])  # if bool\string filed for each parameter num actions= len dict ['true','false']-->len=2

    num_actions_return = num_actions
    num_actions = num_actions - 1  # action dict will start from 0 num_action-1

    for k, v in enumerate(dict):
        x = dict[v]['init_val']
        # stage 1 check if numeric field-if yes convert it to value between 0 to 1
        if type(dict[v]['init_val']) not in [str, bool]:

            min_val = dict[v]['range_val'][0]
            max_val = dict[v]['range_val'][1]
            norm_val = (x - min_val) / (max_val - min_val)

            numeric_param_norm_init_dict[v] = norm_val
            min_max_numeric_field_dict[v] = [min_val, max_val]
            # map parameter with action +0.01,-0.01...
            action_map[num_actions] = [v, 0.1]
            action_map[num_actions - 1] = [v, -0.1]
            # new added
            action_map[num_actions - 2] = [v, 0.01]
            action_map[num_actions - 3] = [v, -0.01]
            num_actions = num_actions - 4

        else:

            # adding all categories and a boolean value to the action_map dictionary so that action_map will include all combinations of categorical parameters
            for k in range(len(dict[v]['range_val'])):
                action_map[num_actions - k] = [v, dict[v]['range_val'][k]]

                # if it is the initial value the user chooses for the categorical parameter, we will put 1 in the value else 0
                if dict[v]['range_val'][k] == x:
                    cat_bool_dict[v, dict[v]['range_val'][k]] = 1
                else:
                    cat_bool_dict[v, dict[v]['range_val'][k]] = 0

            num_actions = num_actions - len(dict[v]['range_val'])

    # reterning the initial dictionary with the values that the user choose
    init_dict = {**numeric_param_norm_init_dict, **cat_bool_dict}

    # start index from 0
    num_actions = num_actions
    return init_dict, numeric_param_norm_init_dict, min_max_numeric_field_dict, cat_bool_dict, action_map, num_actions_return


def change_parameter_to_real_val(obs, min_max_numeric_field_dict):
    dict = obs.copy()
    un_norm_dict = OrderedDict()
    for v in dict.keys():
        # if numeric change to real scale
        if type(v) not in [tuple]:
            y = dict[v]
            min_val = min_max_numeric_field_dict[v][0]
            max_val = min_max_numeric_field_dict[v][1]
            if y > 1:
                y = 1
            if y < 0:
                y = 0
            x = round(((max_val - min_val) * y) + min_val, 3)  # new 25.9.2021 round x to 3 decimal after the dot

            # if real valye of parameter >1 round up
            if x < 1:
                un_norm_dict[v] = x
            else:
                un_norm_dict[v] = int(x)
        # if bool or categorical take value of the parameter to be where 1
        else:
            # unpack tupel
            a, b = v
            if dict[v] == 1:
                un_norm_dict[a] = b

    return un_norm_dict


def get_init_dict(dict):
    init_dict = OrderedDict()
    for v in user_dict.keys():
        init_dict[v] = user_dict[v]['init_val']
    return init_dict


def change_bool_cat_val_by_choosen_action(obs, action_map, action):
    dict = obs.copy()
    for k in dict.keys():
        if type(k) == tuple:
            a, b = k
            if a == action_map[action][0]:
                if b == action_map[action][1]:
                    dict[(a, b)] = 1
                else:
                    dict[(a, b)] = 0
    return dict


def change_next_state_to_real_val(obs, action_map, action, min_max_numeric_field_dict):
    next_state = obs.copy()

    if (type(action_map[action][1]) in [str, bool]):
        next_state = change_bool_cat_val_by_choosen_action(obs, action_map, action)
    else:
        next_state[action_map[action][0]] += action_map[action][1]

    next_state_real_val_t = change_parameter_to_real_val(next_state, min_max_numeric_field_dict)

    return next_state_real_val_t, next_state


def get_aveilable_action_index_and_val(Total_counter, actions, action_map, obs, min_max_numeric_field_dict,
                                       is_random_choice, state_dict):
    global done_all_action_inf
    done_all_action_inf = False
    new_state = False
    stop = False
    a = actions
    next_state = obs.copy()

    if is_random_choice == 'not random choice':
        maxQ = T.max(a).item()

        while (not new_state):  # while didnt found action that will give diffrent hyperparameter combination
            next_state = obs.copy()
            action = T.argmax(a).item()
            maxQ = T.max(a).item()
            next_state_real_val_t, next_state = change_next_state_to_real_val(obs, action_map, action,
                                                                              min_max_numeric_field_dict)

            if next_state_real_val_t in state_dict:  ###check if state allrady exsist if yes change difrrent action
                a[action] = -np.inf
                c = a.cpu().detach().numpy()
                print('Iteration:', Total_counter)  # ,'repeted action Q list',maxQ,'c',c)
                print('-------------------------------------------------------------')
                print('repeted action Q list:')
                print('-------------------------------------------------------------')
                pprint.pprint(maxQ)
                print('-------------------------------------------------------------')
                print('-------------------------------------------------------------')
                print('Q list after removing repeted action:')
                print('-------------------------------------------------------------')
                pprint.pprint(c)
                print('-------------------------------------------------------------')
                new_state = False

            else:
                new_state = True

            if (maxQ == -np.inf):
                print(maxQ, 'should break all actions lead to repeated hyperparam combo')
                new_state = True
                stop = True
                done_all_action_inf = True

        next_state_real_val = next_state_real_val_t
        current_state_for_model_error_checks = change_parameter_to_real_val(obs, min_max_numeric_field_dict)
        return next_state, action, current_state_for_model_error_checks, next_state_real_val



    else:  # if random action
        action_try_list = []
        action = actions  # in random we will have here 1 action
        while (not new_state):
            next_state = obs.copy()
            next_state_real_val_t, next_state = change_next_state_to_real_val(obs, action_map, action,
                                                                              min_max_numeric_field_dict)  # change state with action choosen

            if next_state_real_val_t not in state_dict:
                new_state = True

            else:  ### if state allrady exsist
                action_try_list.append(action)
                print('repeted random action:', action)
                list_withot_actions_tried = np.setdiff1d(list(agent.action_space.keys()),
                                                         action_try_list)  # get list of agent actions that we didnt tried

                if len(list_withot_actions_tried) == 0:
                    print('all random action was tried')
                    done_all_action_inf = True
                    new_state = True
                else:
                    action = random.choice(list_withot_actions_tried)  # choose diffrent random action
                    # print('Iteration:',Total_counter,'new random action: ',action,' still need to try this actions: ', list_withot_actions_tried)
                    print('Iteration:', Total_counter)
                    print('-------------------------------------------------------------')
                    print('repeted action trying new action:')
                    print('-------------------------------------------------------------')
                    pprint.pprint(action)
                    print('-------------------------------------------------------------')
                    print('action lefts after removing repeted actions:')
                    print('-------------------------------------------------------------')
                    pprint.pprint(list_withot_actions_tried)
                    print('-------------------------------------------------------------')
                    new_state = False

        next_state_real_val = next_state_real_val_t
        current_state_for_model_error_checks = change_parameter_to_real_val(obs, min_max_numeric_field_dict)
        return next_state, action, current_state_for_model_error_checks, next_state_real_val


def save_models_for_expirament(time, counter_checking_all_data):
    iter = counter_checking_all_data
    # create path to save the model
    # checkpoint_file_q_eval = os.path.join(agent.q_eval.checkpoint_dir, 'q_eval_'+str(round(iter)))
    # checkpoint_file_q_next = os.path.join(agent.q_next.checkpoint_dir, 'q_next_'+str(round(iter)))
    checkpoint_file_q_eval = os.path.join(agent.q_eval.checkpoint_dir, agent.q_eval.checkpoint_file + str(round(iter)))
    checkpoint_file_q_next = os.path.join(agent.q_next.checkpoint_dir,
                                          agent.q_next.checkpoint_file + str(round(iter)))  # dont need to save?
    # save all models names into list
    models.append(checkpoint_file_q_eval)
    model_num_iter.append(iter)
    runing_times.append(time)

    # network_dict_q_eval[str(round(iter))]=checkpoint_file_q_eval
    # network_dict_q_next[str(round(iter))]=checkpoint_file_q_next
    print('... saving checkpoint ...')
    T.save(agent.q_eval.state_dict(), checkpoint_file_q_eval)
    T.save(agent.q_next.state_dict(), checkpoint_file_q_next)


def save_array_to_drive(Tfname, Tfdata):
    save_path_train = path_pycharm + 'config changing tests/Trials/' + Tfname + '.csv'
    # data=Tfdata
    # # save to csv file
    # savetxt(save_path_train, data, delimiter=',',fmt='%s')
    np.save(save_path_train, Tfdata)

    print('saving array...')


def load_array_from_drive(Tfname):
    save_path_train = path_pycharm + 'config changing tests/Trials/' + Tfname + '.csv'
    # data=loadtxt(save_path_train, delimiter=',')
    data = np.load(save_path_train + '.npy')
    return data


# Read in the hyper-parameters and return a Run namedtuple containing all the
# combinations of hyper-parameters
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs







def init_params():
    global train_result_df, Param_df, chkpt_dir, episode_durations, train_losses, avg_train_losses, best_score, load_checkpoint
    global time_network_dict_q_eval, time_network_dict_q_next, stop, models, model_num_iter, done_all_action_inf, best_search, runing_times
    global init_dict, numeric_param_norm_init_dict, min_max_numeric_field_dict, cat_bool_dict, action_map, num_actions_return
    global episode_size, episodes_number, episod_num_iter, batch_size, early_stopping, num_iteration_checking_all_data, avg_over_iteration
    global Total_counter, Total_iteration, counter, learner_model, search, input_dims, n_actions, n_steps, scores, eps_history, steps_array, init_dict_un_norm, Run_start, last_time, counter_checking_all_data, avg_score

    # Train, Test  = train_test_split(Data,test_size = 0.33, random_state = 42,shuffle=False)

    # Data for saving into scv tables on drive to QA the prosses
    train_result_df = pd.DataFrame(
        columns=['lr', 'episode_num', 'is_random_action', 'eps', 'state', 'action', 'mape_error', 'next_s'])
    Param_df = pd.DataFrame()
    chkpt_dir = path_pycharm + 'models/'
    # '/content/drive/My Drive/MorTeza-main/TEZA/models/'
    # Parameters to save models and compare with other algorithems

    time_network_dict_q_eval = {}
    time_network_dict_q_next = {}
    stop = False
    models = []
    model_num_iter = []
    runing_times = []

    # Parameters for early stopping
    train_losses = []
    avg_train_losses = []

    # Parameters for printing values to QA iterations
    best_search = {}
    episode_durations = []

    ###################################init all parameters############################
    best_score = -np.inf
    # training mode load_checkpoint=False
    load_checkpoint = False
    # change user dict to NN dict =norm numeric+categorical
    init_dict, numeric_param_norm_init_dict, min_max_numeric_field_dict, cat_bool_dict, action_map, num_actions_return = change_numeric_parameter_scal(
        user_dict)

    # Define the number of rows we will take from the data in each episode 1000 unless data set is smaller
    if len(Train) < 1000:
        episode_size = len(Train)
    else:
        episode_size = 1000

    # The number of episodes defines in such a way that we will run 10 times on all rows in the data set
    #episodes_number = 10 * round((len(Train.index) / episode_size))  # To do change to 10
    episodes_number = round((len(Train.index) / episode_size))  # change to 10
    print('episodes_number:', episodes_number)

    # The number of iteration in each episode is defined as : length of train data set/1000
    # in each episode, we are running on 1000 rows from the data set

    # Define the number of rows we will take from the data in each episode 1000 unless data set is smaller
    # episod_num_iter = max(number of hyperparameter*10 ,(len(Train.index)/episode_size))- give chance of at least 10 actions in each epoch
    episod_num_iter = round((len(Train.index) / episode_size))

    if episod_num_iter < len(user_dict) * 10:
        episod_num_iter = len(user_dict) * 10
    else:
        episod_num_iter = round((len(Train.index) / episode_size))

    print('number of iteretion in episode=', episod_num_iter)

    # batch_size:(memory>=batch_size)the agent start learning after 1 episode is completed
    batch_size = episod_num_iter * 5

    # early_stopping-TO DO-check if im using it
    early_stopping = EarlyStopping(patience=episod_num_iter * 3, verbose=True,
                                   delta=0.001)  # אם לא השתפר 5 אפסודים נעצור מתחילים לספור אחרי 50% מכל האיטרציות

    # Define the number of the episode needed reading all rows in data 1 time-and, save the model each time we passed all the data
    # (episodes_number/10 = num episode reading all rows in data 1 time)
    num_iteration_checking_all_data = episod_num_iter * (episodes_number / 10)
    avg_over_iteration = num_iteration_checking_all_data
    Total_counter = 0
    Total_iteration = episodes_number * episod_num_iter

    # Curiosity algorithm parameters
    counter = 0
    # learner_model ='RF_Classifier'#'ExtraTreesClassifier'#'DT'#'RF_Classifier'#'RF_Regressor' #'gbdt'#'RF_Regressor'
    search = init_dict.copy()
    input_dims = len(search)
    n_actions = num_actions_return
    # Plots parameters
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    print('episodes_number:', episodes_number, 'episod_num_iter=', episod_num_iter, 'num_iteration_checking_all_data=',
          num_iteration_checking_all_data, 'Total_iteration=', Total_iteration)
    print('Lerner: ', learner_model)
    print('batch_size: ', batch_size)
    print('init_dict norm', init_dict)
    init_dict_un_norm = get_init_dict(user_dict)
    print('init_dict_un_norm', init_dict_un_norm)

    # for ploting time it takes the algo to run
    Run_start = timeit.default_timer()
    last_time = Run_start
    # num_iteration_checking_all_data
    counter_checking_all_data = 0
    avg_score = 0


def episode_reword(episode_last_error, episode_error):
    episode_reward = episode_last_error - episode_error
    reward_signe = 0
    if episode_reward > 0:
        reward_signe = 1
        # non_linear_episode_reward=(reward_signe)*(episode_reward**2)
    else:
        reward_signe = -1
        # non_linear_episode_reward=(reward_signe)*(episode_reward**1.5)

    non_linear_episode_reward = (reward_signe) * (episode_reward ** 2)
    return non_linear_episode_reward



def get_openML_data_by_task_id(task_id):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(
        "Task {}: number of repeats: {}, number of folds: {}, number of samples {}.".format(
            task_id, n_repeats, n_folds, n_samples,
        )
    )

    train_indices, test_indices = task.get_train_test_split_indices()

    X, y = task.get_X_and_y(dataset_format="dataframe")
    Data = pd.concat([X, y], axis=1)

    Train = Data.iloc[train_indices]
    Test = Data.iloc[test_indices]
    print('Numbe of rows in train:', len(Train))
    return Train, Test

#init_params()
def train_main():
    global train_result_df, Param_df, chkpt_dir, episode_durations, train_losses, avg_train_losses, best_score, load_checkpoint
    global time_network_dict_q_eval, time_network_dict_q_next, stop, models, model_num_iter, done_all_action_inf, best_search, runing_times
    global init_dict, numeric_param_norm_init_dict, min_max_numeric_field_dict, cat_bool_dict, action_map, num_actions_return
    global episode_size, episodes_number, episod_num_iter, batch_size, early_stopping, num_iteration_checking_all_data, avg_over_iteration
    global Total_counter,agent, Total_iteration, counter, learner_model, search, input_dims, n_actions, n_steps, scores, eps_history, steps_array, init_dict_un_norm, Run_start, last_time, counter_checking_all_data, avg_score

    # init_params()
    # start triels
    m = RunManager()
    # get all runs from params using RunBuilder class
    for run in RunBuilder.get_runs(
            params):  # all parameters we want to check add loss function and eps methot and grad ethod

        # init agent parameters for the run-currently just batch size add loss function and gradient
        batch_size = run.batch_size
        # Create new agent global_algo,global_env_name
        agent = DQNAgent(gamma=0.9, epsilon=0.9, lr=0.09,
                         input_dims=(input_dims,),
                         n_actions=n_actions,
                         mem_size=100000,
                         eps_min=0.001,
                         batch_size=batch_size,
                         replace=episod_num_iter,
                         eps_dec=0.0001,
                         chkpt_dir=chkpt_dir,
                         algo=global_algo,
                         env_name=global_env_name,
                         action_space=action_map)

        m.begin_run(run, agent, episode_size)

        if load_checkpoint:
            agent.load_models()

        # Plotting
        fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(episodes_number) + 'Train'
        # creat plot directory

        figure_file = path_pycharm + 'config changing tests/plots/' + fname + '.png'

        global min_errors
        min_errors = 1.0

        for i in range(int(episodes_number)):
            ##########################  episode init params  ##########################

            agent.train_losses = []
            done_all_action_inf = False
            done_episode = False
            obs = search.copy()
            next_obs = search.copy()
            counter = 0
            score = 0
            episode_error = 0
            # Create the episode data
            episode = get_data(episode_size, policy=0, mode='train')
            # Separate the episode data into features and label
            X_episode, y_episode = data_separate(episode)

            stratify_by = y_episode  # None - classification problems use y_episode else none :keep the proportion of y values through the training and test sets
            X_train_main, X_test_main, y_train, y_test = data_split(X_episode, y_episode, stratify_by)  # y_episode
            # each episode, we are starting from the same initial dictionary that the user defined
            init_dict = init_dict_un_norm
            # Initiate the error to initial state parameters error

            # init_dict=change_dictionery_to_exp (init_dict) #only for mnist remove line for other data sets

            episode_last_error = Learner(X_train_main, X_test_main, y_train, y_test, learner_model=learner_model,
                                         parameter_dict=init_dict)
            # episode_last_error=Learner(X_episode,y_episode,learner_model=learner_model,parameter_dict=init_dict)
            # check how much time it takes each episode to run
            episod_start = timeit.default_timer()
            # We save all the combinations of hyperparameters we tried for each epoch to avoid repeating them.
            # We clean it at the beginning of each epoch to an empty array
            state_dict = []

            m.begin_epoch()  # for dashboard

            while not done_episode:

                Total_counter = Total_counter + 1
                counter = counter + 1
                actions, is_random_choice = agent.choose_action(obs, mood='Train')
                if is_random_choice == 'random choice':
                    next_obs, action, current_state_for_model_error_checks, next_state_real_val = get_aveilable_action_index_and_val(
                        Total_counter, actions, action_map, obs, min_max_numeric_field_dict, is_random_choice,
                        state_dict)  # new today
                if is_random_choice == 'not random choice':
                    next_obs, action, current_state_for_model_error_checks, next_state_real_val = get_aveilable_action_index_and_val(
                        Total_counter, actions, action_map, obs, min_max_numeric_field_dict, is_random_choice,
                        state_dict)  # new today
                if done_all_action_inf == False:
                    state_dict.append(current_state_for_model_error_checks)
                    state_dict.append(next_state_real_val)  # remove??

                    #####################################For Fasion Mnist QA Only#################################################
                    current_state_for_model_error_checks['bootstrap'] = False
                    current_state_for_model_error_checks['criterion'] = 'gini'
                    #####################################For Fasion Mnist QA Only#################################################
                    # episode_error=Learner(X_episode,y_episode,learner_model=learner_model,parameter_dict=current_state_for_model_error_checks)

                    episode_error = Learner(X_train_main, X_test_main, y_train, y_test, learner_model=learner_model,
                                            parameter_dict=current_state_for_model_error_checks)
                    # Update reward
                    # episode_reward=episode_last_error-episode_error #
                    episode_reward = episode_reword(episode_last_error, episode_error)  # non linear reword new
                    reward = episode_reward
                    score += reward

                    # agent.q_eval.optimizer.param_groups[0]['lr']
                    new_row = {'lr': agent.lr, 'episode_num': i, 'is_random_action': is_random_choice,
                               'eps': agent.epsilon, 'state': current_state_for_model_error_checks.copy(),
                               'action': action_map[action], 'mape_error': episode_error, 'next_s': next_state_real_val}
                    # append row to the dataframe
                    train_result_df = train_result_df.append(new_row, ignore_index=True)

                    df = pd.DataFrame(current_state_for_model_error_checks, index=[0])
                    df['epoch'] = i
                    df['iter'] = counter
                    df['is_random_choice'] = is_random_choice
                    df['action'] = str(action_map[action])
                    df['error'] = episode_error
                    df['Q if not random else action number'] = str(actions)

                    Param_df = Param_df.append(df, ignore_index=True)

                    if min_errors > episode_error:
                        min_errors = episode_error
                        best_search = current_state_for_model_error_checks.copy()

                    if not load_checkpoint:
                        agent.store_transition(list(obs.values()), action,
                                               reward, list(next_obs.values()), int(done_episode))

                        # was at the end of episode changed it to here
                        agent.learn(Total_counter, Total_iteration)
                        # for tensorboard grafh
                        if len(agent.train_losses) > 0:
                            loss = agent.train_losses[len(agent.train_losses) - 1]  # recored loss for graph
                            m.track_loss(loss)

                        m.track_reward(episode_reward)

                is_done = ((counter == episod_num_iter) or (stop == True) or (
                        done_all_action_inf == True))  # or (early_stopping.early_stop and i > 5)
                done_episode = (is_done)
                obs = next_obs.copy()
                episode_last_error = episode_error
                n_steps += 1

                if avg_score > best_score:
                    best_score = avg_score

                # iteretion time
                now = timeit.default_timer()
                time_passed_last_time = now - last_time
                time_passed = now - Run_start

                # We save the model each time we finish running on all the rows in the data set.
                if (Total_counter % num_iteration_checking_all_data == 0):
                    counter_checking_all_data += 1
                    print('Number of times checking_all_data', counter_checking_all_data)
                    TOTAL_COMBO_all_data = episod_num_iter * counter_checking_all_data
                    print('saving weights', Total_counter)
                    save_models_for_expirament(time_passed, counter_checking_all_data)
                    # save_models_for_expirament(time_passed,Total_counter)
                    last_time = now

                if done_episode:
                    # agent.learn(Total_counter,Total_iteration)
                    m.end_epoch()
                    print('new epoch why? ', '(counter==episod_num_iter): ', (counter == episod_num_iter),
                          ' erly stopping?', stop == True, ' done_all_action_inf? ', done_all_action_inf)
                    episod_end = timeit.default_timer()
                    episod_time = episod_end - episod_start
                    episode_durations.append(episod_time)
                    scores.append(score)
                    steps_array.append(n_steps)
                    avg_score = np.mean(scores[-int(avg_over_iteration):])
                    sum_score = np.sum(scores[:])
                    eps_history.append(agent.epsilon)
                    Tfname = 'Param df_' + agent.algo + '_' + agent.env_name
                    save_path_params = path_pycharm + 'config changing tests/Trials/' + Tfname + '.csv'
                    Param_df.to_csv(save_path_params)
                    print('episod time', episod_time, 'epsilon:', agent.epsilon, 'episode: ', i, 'score: ', score,
                          'best score %.2f' % best_score, 'best_params', best_search, 'min_error', min_errors,
                          current_state_for_model_error_checks)

        time.sleep(100)
        x = [i + 1 for i in range(len(scores))]
        plot_learning_curve(steps_array, scores, eps_history, figure_file)

        # save to drive
        agent.save_models()
        # saving last run into dictionery as well
        counter_checking_all_data += 1
        print('Number of times checking_all_data', counter_checking_all_data)
        save_models_for_expirament(time_passed, counter_checking_all_data)

        # save_models_for_expirament(time_passed,Total_counter)
        last_time = now
        Tfname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + '_eps' + str(
            agent.epsilon) + '_' + 'iteretion_number' + str(Total_counter)
        save_path_train = path_pycharm + 'config changing tests/Trials/' + Tfname + '.csv'
        train_result_df.to_csv(save_path_train)
        save_array_to_drive(agent.algo + '_' + agent.env_name + 'models', models)
        save_array_to_drive(agent.algo + '_' + agent.env_name + 'model_num_iter', model_num_iter)
        m.end_run()
        # save_dictionery?? time_network_dict_q_eval

    # when all runs are done, save results to files
    m.save('results')


if __name__ == '__main__':

   # % load_ext tensorboard , #% tensorboard - -logdir = runs
    global global_algo, global_env_name, path_pycharm, Train, Test, task_id,agent
    path_pycharm = '/Users/zlililaor/Desktop/MorTeza-main/TEZA/' #'/home/mor/project/TEZA/'
    global_algo = 'DQNAgent'
    global_env_name = 'FasionMnist+RFClassifier+epoch num iter 10*hyperparams'

    #Data##########################################################################################################################
    # Fasion mnist
    task_id = 146825
    Train, Test = get_openML_data_by_task_id(task_id)
    #Import class i wrote##########################################################################################################################

    code_path = path_pycharm + 'code'
    sys.path.append(code_path)
    import os.path
    from Agent_change_eps_and_LR_by_iter import DQNAgent
    from utils import plot_learning_curve
    from pytorchtools import EarlyStopping
    from Tensorboard_functions import RunManager
    user_dict = OrderedDict([ ('n_estimators', {'range_val': [2, 500], 'init_val': 250})])
    learner_model = 'RF_Classifier'  # 'ExtraTreesClassifier'#'DT'#'RF_Classifier'#'RF_Regressor' #'gbdt'#'RF_Regressor'
    init_params()
    # for expiraments in tensorboard: put all hyper params into a OrderedDict, easily expandable
    params = OrderedDict(
        # lr = [.01, .001],
        batch_size=[batch_size]
        # shuffle = [False]
    )
    train_main()

