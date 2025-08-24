from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import Random
import collections
from collections import deque, defaultdict
import math
from river import feature_extraction, compose, metrics, drift, naive_bayes, multiclass, utils
from properties import data_folder, graphs_folder
import os
import time

# DESC: Our main efforts towards  optimizing our classifier. A number of parameters are calibrated such as the number
# of adaboost's models, the alpha parameter of Naive Bayes, the size of the inactivity window for every developer and
# the scaling frequency factor that affects the lambda parameter of the adaboost's online adaptation.

def calculate_slope(accuracy_history):
    if len(accuracy_history) < 2:
        return 0
    x = np.arange(len(accuracy_history))
    y = np.array(accuracy_history)
    slope, _ = np.polyfit(x, y, 1)  # Fit a simple line to estimate slope
    return slope

start_time = time.time()

def main(n_models, a_value, w_size, dev_f, reset_f):

    # A flag that indicates whether the model should be reset upon finding a drift
    reset_flag = reset_f
    # Create an interactive plot - switch interactive boolean to make it one-off
    interactive = False
    # Declare adaboost's number of base classifiers
    number_models = n_models
    # Set a value related to a random number generator
    seed = 42
    # Set Naive Bayes alpha parameter
    alpha_val = a_value
    # Set the time window (in days format)
    window = w_size
    # Set the maximum length of the queue tha will hold the latest accuracy results for the slope analysis
    max_length = 10
    # Set the impact factor of the developer-frequency analysis
    dev_freq_factor = dev_f
    # Set the max value of the lambda_poisson parameter for the adaboost algorithm
    lambda_max_value = 5
    # Set the minimum steep_threshold of the slope in order to determine if the models will be reset
    steep_threshold = 0.0002
    # Set the delta parameter for the ADWIN drift detector
    delta_val = 0.00001


    # Lists that hold the project's performance results for each metric
    metric_table = []
    metric_table_macro_f1 = []
    metric_table_macro_recall = []

    for f in os.listdir(data_folder):
        if f.startswith("2_"):

            # The implementation of the concept drift was acquired directly from the following example here:
            # https://riverml.xyz/latest/introduction/getting-started/concept-drift-detection/
            drift_detector = drift.ADWIN(delta=delta_val)
            drifts = []

            accuracy_history = deque(maxlen=max_length)

            # Read data for a project
            data = pd.read_csv(os.path.join(data_folder, f))
            project_name = data["project_name"].iloc[0]
            # print("Processing : ", project_name)

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

            fig, ax = plt.subplots(figsize=(24,12))
            # Set the figure's window title
            fig.canvas.manager.set_window_title(project_name)
            fig.canvas.mpl_connect('close_event', exitplot)
            # Set the names for the x and y axes of the plots
            plt.xlabel("Number of Instances", fontsize = 18)
            plt.ylabel("Moving Average Accuracy", fontsize = 18)
            # plt.tight_layout()
            # Save the figures
            str_save_int = os.path.abspath(graphs_folder + project_name) + "/" + project_name + "-" + "interactive.png"
            str_save_off = os.path.abspath(graphs_folder + project_name) + "/" + project_name + "-" + "complete.png"

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
            metric_macro_f1 = metrics.MacroF1()
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

                if dev_freq_factor != 0:
                    lambda_poisson = 1 + weight_dev_freq
                else:
                    lambda_poisson = 1

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

                metric.update(y, final_prediction)
                if final_prediction is not None:
                    # Update the selected metric values
                    metric_macro_f1.update(y, final_prediction)
                    metric_macro_recall.update(y, final_prediction)


                result = metric.get()
                results.append(result)
                # Scan for drifts
                drift_detector.update(result)
                accuracy_history.append(result)


                if drift_detector.drift_detected:
                    # print(f"Change detected at index {i}")

                    slope = calculate_slope(accuracy_history)
                    # print("slope is: ", slope)

                    if slope < 0 and reset_flag:

                        if abs(slope) >= steep_threshold:
                            drifts.append([i, "red"])
                            # print("Reset the models")
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

            # print(metric)
            # print(metric_macro_f1)
            # print(metric_macro_recall)

            metric_table.append(metric.get())
            metric_table_macro_f1.append(metric_macro_f1.get())
            metric_table_macro_recall.append(metric_macro_recall.get())

            # print(metric_table)

            # Plot the final metric plot
            if interactive:
                plt.ioff()

            # Case handling where Interactive == False
            fig_2, ax_2 = plt.subplots(figsize=(12,8))
            fig_2.canvas.manager.set_window_title(project_name)
            plt.xlabel("Number of Instances", fontsize=18, labelpad=15, fontweight="bold")
            plt.ylabel("Moving Average Accuracy", fontsize=18, labelpad=15, fontweight="bold")
            # plt.title(project_name, fontsize=18, pad=15, fontweight="bold")
            plt.tight_layout()

            ax_2.tick_params(axis='both', which='major', labelsize=14)  # Adjust fontsize here
            for tick in ax_2.get_xticklabels() + ax_2.get_yticklabels():
                tick.set_fontweight('bold')  # Set tick labels to bold


            plt.plot(results, 'b')

            # print("Interactive mode is done")
            if drifts is not None:
                for drift_detected in drifts:
                    # Place a line exactly where the drift was found
                    if interactive:
                        ax.axvline(drift_detected[0], color=drift_detected[1])
                    ax_2.axvline(drift_detected[0], color=drift_detected[1])

            if interactive:
                fig.savefig(str_save_int)
            # plt.show()
            fig_2.savefig(str_save_off)

    # print("Average Accuracy across all projects: ", sum(metric_table) / len(metric_table))
    # print("Average macro f1 across all projects: ", sum(metric_table_macro_f1) / len(metric_table_macro_f1))
    # print("Average macro recall across all projects: ", sum(metric_table_macro_recall) / len(metric_table_macro_recall))

    return sum(metric_table) / len(metric_table)

def plot_results(x_values, y_values, f):

    x_axis = x_values
    y = y_values
    flag = f

    if flag == 1:
        x_title = "Number of Base Models"
        y_title = "Average accuracy across all projects (%)"
        graph_title = "Optimal number of base models for Adaboost"
        str_save_off = os.path.abspath(graphs_folder) + "/" + "5.5 Optimal number of base models for Adaboost" + ".png"

    elif flag == 2:
        x_title = "alpha"
        y_title = "Average accuracy across all projects (%)"
        graph_title = "Optimal value for the alpha parameter of Naive Bayes"
        str_save_off = os.path.abspath(graphs_folder) + "/" + "5.5 Optimal value for the alpha parameter of Naive Bayes" + ".png"

    elif flag == 3:
        x_title = "Window Size (in days)"
        y_title = "Average accuracy across all projects (%)"
        graph_title = "Optimal Window Size for Inactivity Analysis"
        str_save_off = os.path.abspath(graphs_folder) + "/" + "5.5 Optimal Window Size for Inactivity Analysis" + ".png"

    elif flag == 4:
        x_title = "Developer Frequency Scaling Factor (SF)"
        y_title = "Average accuracy across all projects (%)"
        graph_title = "Optimal Developer Frequency Scaling Factor"
        str_save_off = os.path.abspath(graphs_folder) + "/" + "5.5 Optimal Developer Frequency Scaling Factor" + ".png"

    y_percentage = []

    for i in y:
        y_percentage.append(round(i * 100, 2))

    plt.figure(figsize=(16, 10))

    plt.rcParams.update({'font.size': 16})  # You can adjust the size here

    # Find the index of the maximum y value
    max_value = max(y_percentage)
    min_value = min(y_percentage)
    max_index = y_percentage.index(max(y_percentage))
    x_max = x_axis[max_index]

    # Add annotation for the maximum value
    plt.annotate(f'({y_percentage[max_index]}%)',
                 (x_axis[max_index], y_percentage[max_index]),
                 textcoords="offset points",
                 xytext=(0, 10), ha='center')

    plt.plot(x_axis, y_percentage, color="black", marker="o", linestyle='-', markersize=10, linewidth=2.5)

    plt.xlabel(x_title, fontweight="bold", fontsize=16, labelpad=15)
    plt.ylabel(y_title, fontweight="bold", fontsize=16, labelpad=15)
    plt.title(graph_title, fontweight="bold", fontsize=16, pad=15)

    if flag == 2:
        plt.xscale('log')
        plt.xticks(x_axis, labels=[str(x) for x in x_axis])

    if flag == 4:
        plt.xticks(np.arange(0.05, 0.55, 0.05))  # Adjust the step size as needed

    # Set bold fontweight for x and y tick labels
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.tight_layout()
    plt.gca().set_facecolor('#e6f2ff')  # Soft blue color
    plt.grid(True, color='white')

    plt.savefig(str_save_off)
    # plt.show()


max_acc_n_models = 0
max_acc_w_size = 0
max_acc_a_value = 0
max_acc_dev_f = 0
max_acc_reset_f = 0

max_acc = 0
max_acc_temp = 0
x_axis_values = []
y_axis_values = []

for n_models in range(5, 36, 5):
    print("Number of models: ", n_models)
    x_axis_values.append(n_models)

    max_acc_temp = main(n_models, 0.1, 0, 0 , 0)

    y_axis_values.append(max_acc_temp)

    if max_acc_temp > max_acc:
        max_acc = max_acc_temp
        max_acc_n_models = n_models

print("Optimized Number of Models : ", max_acc_n_models)
print("Accuracy with optimized number of models is: ", max_acc)

plot_results(x_axis_values, y_axis_values,1)

max_acc = 0
max_acc_temp = 0
x_axis_values = []
y_axis_values = []

for a_value in [0.0001, 0.001, 0.01, 0.1, 1]:
    print("alpha value is: ", a_value)
    x_axis_values.append(a_value)

    max_acc_temp = main(max_acc_n_models, a_value, 0, 0, 0)

    y_axis_values.append(max_acc_temp)

    if max_acc_temp > max_acc:
        max_acc = max_acc_temp
        max_acc_a_value = a_value

print("Optimized alpha value is: ", max_acc_a_value)
print("Accuracy with optimized alpha_value is : ", max_acc)

plot_results(x_axis_values, y_axis_values,2)

max_acc = 0
max_acc_temp = 0
x_axis_values = []
y_axis_values = []

for w_size in range(30, 181, 10):
    print("window size is: ", w_size)
    x_axis_values.append(w_size)

    max_acc_temp = main(max_acc_n_models, max_acc_a_value, w_size, 0, 0)

    y_axis_values.append(max_acc_temp)

    if max_acc_temp > max_acc:
        max_acc = max_acc_temp
        max_acc_w_size = w_size

print("Optimized window size is: ", max_acc_w_size)
print("Accuracy with optimized window size is : ", max_acc)

plot_results(x_axis_values, y_axis_values,3)

max_acc = 0
max_acc_temp = 0
x_axis_values = []
y_axis_values = []

for dev_f in np.arange(0, 0.51, 0.05):
    print("dev freq is : ", dev_f)
    x_axis_values.append(dev_f)

    max_acc_temp = main(max_acc_n_models, max_acc_a_value, max_acc_w_size, dev_f , 0)
    y_axis_values.append(max_acc_temp)

    if max_acc_temp > max_acc:
        max_acc = max_acc_temp
        max_acc_dev_f = dev_f

print("Optimized dev freq is: ", max_acc_dev_f)
print("Accuracy with optimized dev freq is : ", max_acc)

plot_results(x_axis_values, y_axis_values,4)

max_acc = main(max_acc_n_models, max_acc_a_value, max_acc_w_size, max_acc_dev_f , 1) # enable reset for models

print("max acc with reset is: ")
print(max_acc)

print("--- %s seconds ---" % (time.time() - start_time))