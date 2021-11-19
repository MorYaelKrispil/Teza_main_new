
from IPython.display import display
from torch.utils.tensorboard import SummaryWriter
import json
from IPython.display import clear_output
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import numpy as np
from collections import OrderedDict



class RunManager():
    def __init__(self):

        # tracking every epoch count, loss, accuracy, time
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_reward = 0
        self.epoch_start_time = None

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        # record model, agent and TensorBoard
        self.network = None
        self.agent = None
        self.tb = None
        self.accumulated_loss = []
        self.accumulated_loss_multply_by_batch_size = []

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
    def begin_run(self, run, agent, episode_size):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = agent.q_eval
        self.agent = agent
        self.episode_size = episode_size
        self.accumulated_loss = []
        self.accumulated_loss_multply_by_batch_size = []
        # self.log_dir = 'runs/'+agent.env_name
        # self.tb = SummaryWriter(log_dir=self.log_dir+ f'-{run}')#???
        # self.tb = SummaryWriter(log_dir=self.log_dir,comment=f'-{run}')
        self.tb = SummaryWriter(comment=f'-{run}' + agent.env_name)

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self):
        self.tb.close()
        self.tb.flush()
        self.epoch_count = 0

    # zero epoch count, loss, accuracy,
    def begin_epoch(self):

        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_reward = 0
        self.accumulated_loss = []
        self.accumulated_loss_multply_by_batch_size = []

        # self.epoch_num_correct = 0

    #
    def end_epoch(self):
        # calculate epoch duration and run duration(accumulate)

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # record epoch loss and reward

        reward = self.epoch_reward
        # Record epoch reward to TensorBoard
        self.tb.add_scalar('Reward', reward, self.epoch_count)

        # loss = self.epoch_loss / self.episode_size #len(self.loader.dataset)

        if (len(self.accumulated_loss)) > 0:

            loss = np.mean(self.accumulated_loss)
            loss_multply_by_batch_size = np.mean(self.accumulated_loss_multply_by_batch_size)
            # Record epoch loss to TensorBoard
            self.tb.add_scalar('Loss', loss, self.epoch_count)
            self.tb.add_scalar('Loss * batch size', loss_multply_by_batch_size, self.epoch_count)

            for name, param in self.network.named_parameters():
                self.tb.add_histogram(name, param, self.epoch_count)
                self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["reward"] = reward
        # results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        if (len(self.accumulated_loss)) > 0:
            results["loss"] = loss
            results["loss_multply_by_batch_size"] = loss_multply_by_batch_size

        # Record hyper-params into 'results'
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # display epoch information and show progress
        clear_output(wait=True)
        display(df)

    # accumulate loss of batch into entire epoch loss
    def track_loss(self, loss):
        # multiply batch size so variety of batch sizes can be compared
        self.epoch_loss += loss * self.agent.batch_size
        self.accumulated_loss.append(loss)  # loss*batch size?
        self.accumulated_loss_multply_by_batch_size.append(loss * self.agent.batch_size)

    def track_reward(self, reward):
        self.epoch_reward += reward

    # save end results of all runs into csv, json for further a
    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns',
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

