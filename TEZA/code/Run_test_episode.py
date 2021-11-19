import timeit
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch as T
import warnings
import numpy as np
from collections import OrderedDict
warnings.filterwarnings("ignore", category=UserWarning)

class RunAgent_episode():

    def data_split( self,X, y, stratify_by):
        global X_train_main
        global X_test_main
        global y_train
        global y_test
        #scaler = StandardScaler()
        X_train_main, X_test_main, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,
                                                                      stratify=stratify_by)

        return X_train_main, X_test_main, y_train, y_test

    # Function that create the episode data - sample randomaly
    def get_data(self, episode_size, policy, mode):
        global dataset, Train
        if mode == 'train':
            dataset = self.Train.sample(n=episode_size, replace=True)
        else:
            dataset = self.Train
        return dataset

    # Function that separate the episode data into features and label
    def data_separate(self, dataset):
        global X
        global y
        X = dataset.iloc[:, 0:dataset.shape[1] - 1]  # all rows, all the features and no labels
        y = dataset.iloc[:, -1]  # all rows, label only
        return X, y

    def change_next_state_to_real_val(self,obs, action_map, action, min_max_numeric_field_dict):
        next_state = obs.copy()

        if (type(action_map[action][1]) in [str, bool]):
            next_state = self.change_bool_cat_val_by_choosen_action(obs, action_map, action)
        else:
            next_state[action_map[action][0]] += action_map[action][1]

        next_state_real_val_t = self.change_parameter_to_real_val(next_state, min_max_numeric_field_dict)

        return next_state_real_val_t, next_state

    def change_bool_cat_val_by_choosen_action(self,obs, action_map, action):
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

    def change_parameter_to_real_val(self,obs, min_max_numeric_field_dict):
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

    def get_aveilable_action_index_and_val_Test(self, Total_counter, actions, action_map, obs,
                                                min_max_numeric_field_dict,
                                                is_random_choice, state_dict):

        self.done_all_action_inf = False
        new_state = False
        stop = False
        a = actions
        next_state = obs.copy()
        repeted_action = False
        maxQ = T.max(a).item()

        while (not new_state):  # while didnt found action that will give diffrent hyperparameter combination
            next_state = obs.copy()
            action = T.argmax(a).item()
            maxQ = T.max(a).item()
            next_state_real_val_t, next_state = self.change_next_state_to_real_val(obs, action_map, action,
                                                                                   min_max_numeric_field_dict)

            if next_state_real_val_t in self.state_dict:  ###check if state allrady exsist if yes change difrrent action
                a[action] = -np.inf
                c = a.cpu().detach().numpy()
                print('Iteration:', Total_counter)  # ,'repeted action Q list',maxQ,'c',c)
                print('-------------------------------------------------------------')
                print('repeted action')
                print('-------------------------------------------------------------')

                new_state = False
                repeted_action = True

            else:
                new_state = True

            if (maxQ == -np.inf):
                print(maxQ, 'should break all actions lead to repeated hyperparam combo')
                new_state = True
                stop = True
                done_all_action_inf = True

        next_state_real_val = next_state_real_val_t
        current_state_for_model_error_checks = self.change_parameter_to_real_val(obs, min_max_numeric_field_dict)
        return repeted_action, maxQ, next_state, action, current_state_for_model_error_checks, next_state_real_val

    def init_episode(self, search, is_stratify_by, init_dict_un_norm, data_num_rows_for_test, Train):

        self.done_all_action_inf = False
        self.done_episode = False
        self.obs = search.copy()
        self.next_obs = search.copy()
        self.counter = 0
        self.score = 0
        self.episode_error = 0
        self.Total_counter = 0
        self.episod_start = timeit.default_timer()
        self.state_dict = []
        self.is_stratify_by = is_stratify_by
        self.init_dict = init_dict_un_norm
        self.data_num_rows_for_test = data_num_rows_for_test
        self.Train = Train

        # Create the episode data
        episode = self.get_data(self.data_num_rows_for_test, policy=0, mode='train')  # change to 1000
        # Separate the episode data into features and label
        X_episode, y_episode = self.data_separate(episode)
        stratify_by = None
        if self.is_stratify_by == True:
            stratify_by = y_episode
        X_train_main, X_test_main, y_train, y_test = self.data_split(X_episode, y_episode, stratify_by)
        return X_train_main, X_test_main, y_train, y_test

    def close_episode(self, save_path_test, test_result_df, current_state_for_model_error_checks, i,
                      Best_param_train_in_iter,
                      maxQ, episod_num_iter):  # i=o for last network!! to do

        self.episod_num_iter=episod_num_iter
        test_result_df.to_csv(save_path_test)
        # insert hyperparameter into list with triel and value
        best_param = current_state_for_model_error_checks
        Best_param_train_in_iter[i] = best_param
        print('best_param', best_param)
        print(Best_param_train_in_iter)
        print('done reasone: ', 'counter? ', (self.counter == self.episod_num_iter), 'done_all_action_inf? ',
              (self.done_all_action_inf == True), 'maxQ<0 ? ', (maxQ < 0))
        print(tabulate(test_result_df, headers='keys', tablefmt='psql'))
        dictionary = Best_param_train_in_iter

        return dictionary
        # i=o for last network!! to do

    def Run_episode(self, agent, action_map, min_max_numeric_field_dict):
        self.Total_counter = self.Total_counter + 1
        self.counter = self.counter + 1
        actions, is_random_choice = agent.choose_action(self.obs, mood='Test')
        repeted_action, maxQ, self.next_obs, action, current_state_for_model_error_checks, next_state_real_val = self.get_aveilable_action_index_and_val_Test(
            self.Total_counter, actions, action_map, self.obs, min_max_numeric_field_dict, is_random_choice,
            self.state_dict)  # new today
        self.state_dict.append(current_state_for_model_error_checks)
        self.state_dict.append(next_state_real_val)
        return repeted_action, maxQ, self.next_obs, action, current_state_for_model_error_checks, next_state_real_val
