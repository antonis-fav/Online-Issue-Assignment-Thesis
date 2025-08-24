import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from properties import data_folder, graphs_folder, all_issues_data

# Description: Graph the assigned issues for every selected project. We graph the assigned issues before and after
# applying the preprocessing techniques. This will give us an overview of the data.

import time

start_time = time.time()

# Read data for a project
for f in os.listdir(data_folder):
    if f.startswith("2_"):

        df = pd.read_csv(os.path.join(data_folder, f))
        project_name = df["project_name"].iloc[0]
        print("Processing project: ", project_name)

        for p in os.listdir(os.path.join(data_folder, all_issues_data)):
            if project_name in p:
                df_all = pd.read_csv(os.path.join(data_folder, all_issues_data, p))

        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)

        # Sort the dataframe by the created_date feature
        df = df.sort_values("created_date", ascending=True)
        # Reset the index of the dataframe
        df = df.reset_index()

        df_all = df_all.sort_values("created_date", ascending=True)
        df_all = df_all.reset_index()

        # Covert the creation date of issues into toordinal format
        df["created_date"] = pd.to_datetime(df["created_date"]).apply(lambda x: x.toordinal())
        df_all["created_date"] = pd.to_datetime(df_all["created_date"]).apply(lambda x: x.toordinal())

        # Mark the earliest date in the observations
        date = df_all["created_date"].iloc[0]

        results = []
        results_all = []
        counter_prep = 0
        counter_all = 0
        # Set the time window (days format)
        window_days = 30

        for i, row in df.iterrows():
            if row["created_date"] <= date + window_days:
                counter_prep += 1
            else:
                results.append(counter_prep)
                # Don't forget to process the current observation as it doesn't belong to the previous time window
                counter_prep = 1
                date += window_days

        # Reset the time-counter
        date = df_all["created_date"].iloc[0]

        for j, row in df_all.iterrows():
            if row["created_date"] <= date + window_days:
                counter_all += 1
            else:
                results_all.append(counter_all)
                # Don't forget to process the current observation as it doesn't belong to the previous time window
                counter_all = 1
                date += window_days

        fig = plt.gcf()
        # Set figure's window name
        fig.canvas.manager.set_window_title(project_name)
        plt.plot(results, label='Preprocessing', linestyle='-', color='blue')
        plt.plot(results_all, label='No Preprocessing', linestyle='--', color='red')

        # Set the title and labels of the plot
        plt.title("Impact of preprocessing")
        plt.xlabel('Months')
        plt.ylabel('Assigned Issues')
        # Add a grid
        plt.grid(True)
        # Add a legend
        plt.legend()
        # Show the plot
        # plt.show()

        str_save = os.path.abspath(graphs_folder + project_name) + "/" + project_name + "_overview"
        fig.savefig(str_save)
        plt.clf()

print("--- %s seconds ---" % (time.time() - start_time))




