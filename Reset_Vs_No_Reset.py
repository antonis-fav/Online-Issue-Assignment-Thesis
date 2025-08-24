from __future__ import annotations

from os.path import exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import Random
import collections
from collections import deque, defaultdict
import math
from river import feature_extraction, compose, metrics, drift, naive_bayes, ensemble, multiclass, base, utils
from properties import data_folder, graphs_folder
import os
import time


### *** DESC : A script to graph result where reset is applied and where is not.

def calculate_slope(accuracy_history):
    if len(accuracy_history) < 2:
        return 0
    x = np.arange(len(accuracy_history))
    y = np.array(accuracy_history)
    slope, _ = np.polyfit(x, y, 1)  # Fit a simple line to estimate slope
    return slope


fig_2, ax_2 = plt.subplots(figsize=(12, 8))

reset_values = [1, 0]

for reset_flag in reset_values:

    start_time = time.time()

    # A flag that indicates whether the model should be reset upon finding a drift
    # reset_flag = 1
    # Create an interactive plot - switch interactive boolean to make it one-off
    interactive = False
    # Declare adaboost's number of base classifiers
    number_models = 30
    # Set a value related to a random number generator
    seed = 42
    # Set Naive Bayes alpha parameter
    alpha_val = 0.1
    # Set the time window (in days format)
    window = 90
    # Set the maximum length of the queue tha will hold the latest accuracy results for the slope analysis
    max_length = 10
    # Set the impact factor of the developer-frequency analysis
    dev_freq_factor = 0.15
    # Set the max value of the lambda_poisson parameter for the adaboost algorithm
    lambda_max_value = 5
    # Set the minimum steep_threshold of the slope in order to determine if the models will be reset
    steep_threshold = 0.0002
    # Set the delta parameter for the ADWIN drift detector
    delta_val = 0.00001

    # Lists that hold the project's performance results for each metric
    metric_table = []
    metric_table_micro_f1 = []
    metric_table_macro_f1 = []
    metric_table_micro_recall = []
    metric_table_macro_recall = []

    for f in os.listdir(data_folder):
        if f.startswith("2_FLINK"):

            # The implementation of the concept drift was acquired directly from the following example here:
            # https://riverml.xyz/latest/introduction/getting-started/concept-drift-detection/
            drift_detector = drift.ADWIN(delta=delta_val)
            drifts = []

            accuracy_history = deque(maxlen=max_length)

            # Read data for a project
            data = pd.read_csv(os.path.join(data_folder, f))
            project_name = data["project_name"].iloc[0]
            print("Processing : ", project_name)

            # Create a separate folder for each project's graphs
            if not os.path.exists(os.path.join(graphs_folder, project_name)):
                os.makedirs(graphs_folder + project_name)

            # Reverse the order of the rows in the dataframe
            data = data.iloc[::-1]
            data = data.reset_index()

            # Convert the created_date variable of every issue into it's toordinal form
            data["created_date"] = pd.to_datetime(data["created_date"])
            data["created_date"] = data["created_date"].map(lambda x: x.toordinal())
            # Sort the data by ascending chronological order
            data = data.sort_values(by='created_date')


            def exitplot(_):
                global stop
                stop = True

            fig, ax = plt.subplots()
            # Set the figure's window title
            fig.canvas.manager.set_window_title(project_name)
            fig.canvas.mpl_connect('close_event', exitplot)
            # Set the names for the x and y axes of the plots
            plt.xlabel("Number of Instances")
            plt.ylabel("Moving Average Accuracy")
            # plt.tight_layout()
            # Save the figures
            str_save_int = os.path.abspath(graphs_folder + project_name) + "/" + project_name + "-" + "interactive.png"
            str_save_off = os.path.abspath(graphs_folder + project_name) + "/" + project_name + "-" + "Reset_Vs_No_reset" + ".png"

            if interactive:
                plt.ion()
                plt.show()
            stop = False

            # Perfom TFIDF on the text features of the dataset
            transformer = compose.TransformerUnion(
                ("summary", feature_extraction.TFIDF(on="summary")),
                ("description", feature_extraction.TFIDF(on="description")),
                ("labels", feature_extraction.TFIDF(on="labels")),
                ("components_name", feature_extraction.TFIDF(on="components_name")),
                ("priority_name", feature_extraction.TFIDF(on="priority_name")),
                ("issue_type_name", feature_extraction.TFIDF(on="issue_type_name")),

            )
            # Set the baseline model for adaboost
            my_model = multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=alpha_val))

            # Use a metric for evaluating online the model
            metric = metrics.Accuracy()
            metric_micro_f1 = metrics.MicroF1()
            metric_macro_f1 = metrics.MacroF1()
            metric_micro_recall = metrics.MicroRecall()
            metric_macro_recall = metrics.MacroRecall()
            results = []
            #The dictionary that will contain each one of the n_models of adaboost
            models = {}

            # The ._rng of the river's module Adaboost classifier is located inside the file base.ensemble.py
            randomness = Random(seed)

            for j in range(0, number_models):
                models[j] = my_model.clone()

            wrong_weight: collections.defaultdict = collections.defaultdict(int)
            correct_weight: collections.defaultdict = collections.defaultdict(int)

            count_classes = defaultdict(int)
            # Declare a queue that will hold the latest issues within a window time frame
            queue = deque()
            # Declare a dictionary where the frequency of issue assignment of each developer is stored
            dev_freq = {}

            # Iterate row-wise through all the issues inside the dataframe
            for i, row in data.iterrows():

                x, y = row, row["assignee"]

                # Remove all the keys/value pairs that won't be features in our model
                for key in ["index", "id", "project_name", "assignee"]:
                    x.pop(key)

                transformer.learn_one(x)
                x = transformer.transform_one(x)

                y_proba = collections.Counter()

                for counter in range(0, number_models):
                    epsilon = wrong_weight[counter] + 1e-16
                    epsilon /= (correct_weight[counter] + wrong_weight[counter]) + 1e-16
                    if epsilon == 0 or epsilon > 0.5:
                        model_weight = 1.0
                    else:
                        beta_inv = (1 - epsilon) / epsilon
                        model_weight = math.log(beta_inv) if beta_inv != 0 else 0

                    predictions = models[counter].predict_proba_one(x)
                    utils.norm.scale_values_in_dict(predictions, model_weight, inplace=True)
                    y_proba.update(predictions)
                utils.norm.normalize_values_in_dict(y_proba, inplace=True)

                index = -1


                if len(queue) != 0:
                    for idx in range(len(queue)):
                        # print("difference is: ", (x["created_date"] - queue[idx][1]))

                        if abs(row["created_date"] - queue[idx][1]) < window:
                            index = idx
                            # print(index)
                            break

                    if index == -1:
                        queue.clear()
                    else:
                        for _ in range(index):
                            queue.popleft()

                        for key in y_proba.keys():
                            count_classes[key] = 0

                        for item in queue:
                            for key in y_proba.keys():
                                if key in item:
                                    count_classes[key] += 1

                        for key in y_proba.keys():
                            if count_classes[key] == 0:
                                y_proba[key] = 0


                if len(y_proba) != 0:
                    final_prediction = max(y_proba, key=y_proba.get)
                else:
                    final_prediction = None

                if y not in dev_freq.keys():
                    dev_freq[y] = 1
                else:
                    dev_freq[y] += 1

                weight_dev_freq = (sum(dev_freq.values()) / dev_freq[y]) * dev_freq_factor

                if weight_dev_freq > lambda_max_value:
                    weight_dev_freq = lambda_max_value

                lambda_poisson = 1 + weight_dev_freq

                for counter in range(0, number_models):
                    for _ in range(utils.random.poisson(lambda_poisson, randomness)):
                        models[counter].learn_one(x, y)

                    if models[counter].predict_one(x) == y:
                        # print("correct guess ***")
                        correct_weight[counter] += lambda_poisson
                        lambda_poisson *= (correct_weight[counter] + wrong_weight[counter]) / (
                                2 * correct_weight[counter]
                        )

                    else:
                        wrong_weight[counter] += lambda_poisson
                        lambda_poisson *= (correct_weight[counter] + wrong_weight[counter]) / (
                                2 * wrong_weight[counter]
                        )
                queue.append((y, row["created_date"]))

                # Update the selected metric values
                metric.update(y, final_prediction)
                metric_micro_f1.update(y, final_prediction)
                metric_macro_f1.update(y, final_prediction)
                metric_micro_recall.update(y, final_prediction)
                metric_macro_recall.update(y, final_prediction)

                result = metric.get()
                results.append(result)
                # Scan for drifts
                drift_detector.update(result)
                accuracy_history.append(result)

                if drift_detector.drift_detected:
                    print(f"Change detected at index {i}")

                    slope = calculate_slope(accuracy_history)
                    print("slope is: ", slope)

                    if slope < 0 and reset_flag:

                        if abs(slope) >= steep_threshold:
                            drifts.append([i, "red"])
                            print("Reset the models")
                            for j in range(0, number_models):
                                models[j] = my_model.clone()
                        else:
                            drifts.append([i, "black"])
                    else:
                        drifts.append([i, "black"])

                # Plot the data (if interactive mode)
                if i % 30 == 0 and interactive:
                    plt.plot(results, 'blue')
                    plt.title("Instance %05d" % i)
                    plt.pause(0.015)
                if stop:
                    break

            print(metric)
            print(metric_micro_f1)
            print(metric_macro_f1)
            print(metric_micro_recall)
            print(metric_macro_recall)

            metric_table.append(metric.get())
            metric_table_micro_f1.append(metric_micro_f1.get())
            metric_table_macro_f1.append(metric_macro_f1.get())
            metric_table_micro_recall.append(metric_micro_recall.get())
            metric_table_macro_recall.append(metric_macro_recall.get())

            # print(metric_table)

            # Plot the final metric plot
            if interactive:
                plt.ioff()

            # Case handling where Interactive == False



            if reset_flag == 0:
                ax_2.plot(results, label="No_Reset", color='b')
            else:
                ax_2.plot(results, label="Reset" ,color='green')

            # print("Interactive mode is done")
            if drifts is not None:
                for drift_detected in drifts:
                    # Place a line exactly where the drift was found
                    if interactive:
                        ax.axvline(drift_detected[0], color=drift_detected[1])

                    if reset_flag:
                        ax_2.axvline(drift_detected[0], color=drift_detected[1])

            if interactive:
                fig.savefig(str_save_int)



    print("Average Accuracy across all projects: ", sum(metric_table) / len(metric_table))
    print("Average micro f1 across all projects: ", sum(metric_table_micro_f1) / len(metric_table_micro_f1))
    print("Average macro f1 across all projects: ", sum(metric_table_macro_f1) / len(metric_table_macro_f1))
    print("Average micro recall across all projects: ", sum(metric_table_micro_recall) / len(metric_table_micro_recall))
    print("Average macro recall across all projects: ", sum(metric_table_macro_recall) / len(metric_table_macro_recall))

    print("--- %s seconds ---" % (time.time() - start_time))

fig_2.canvas.manager.set_window_title(project_name)
ax_2.set_xlabel("Number of Instances", fontsize=18, labelpad=15, fontweight="bold")
ax_2.set_ylabel("Average Accuracy", fontsize=18, labelpad=15, fontweight="bold")
title = "Reset Vs No Reset for project " + project_name
ax_2.set_title(title, fontsize=18, pad=15, fontweight="bold")
# fig_2.tight_layout()

legend = ax_2.legend(prop={'size': 18, 'weight': 'bold'}, loc='lower right')  # Use `prop` to set size and weight of legend text


# Make the tick labels on both axes bold by accessing them directly
for tick in ax_2.get_xticklabels():
    tick.set_fontsize(16)
    tick.set_fontweight('bold')

for tick in ax_2.get_yticklabels():
    tick.set_fontsize(16)
    tick.set_fontweight('bold')

fig_2.savefig(str_save_off)
plt.show()