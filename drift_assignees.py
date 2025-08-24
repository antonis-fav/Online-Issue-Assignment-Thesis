import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from properties import data_folder, graphs_folder
import time

# DESC: Plot for every developer the number of issues he takes on within a window time frame (here the time window is 30 days)

start_time = time.time()

# Read data for a project
for f in os.listdir(data_folder):
    if f.startswith("2_"):

        data = pd.read_csv(os.path.join(data_folder, f))
        project_name = data["project_name"].iloc[0]
        print("Processing project: ", project_name)

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

        # Mark the earliest date in the observations
        date = data["created_date"].iloc[0]

        uniq_assignees = len(data["assignee"].unique())
        assignee_count = [0] * uniq_assignees
        results = []
        window_days = 30

        for i in range(0, uniq_assignees):
            col = []
            results.append(col)

        # Since the minimum number of assignees for each project is 5, every row of the figure will contain 4 subplots
        if uniq_assignees > 20:
            n_rows = (uniq_assignees / 6)
            if (uniq_assignees % 6) != 0:
                n_rows += 1
            n_cols = 6
        else:
            n_rows = (uniq_assignees / 4)
            if (uniq_assignees % 4) != 0:
                n_rows += 1
            n_cols = 4

        soft_blue = (0.9, 0.95, 1)  # Light blue color
        # Declare the parameters for the figure
        fig, axes = plt.subplots(int(n_rows), n_cols, figsize=(20, 10))
        # Flatten the 2D array axes for easier handling
        axes = axes.flatten()
        # See the current project's name as the name of the figure's window
        fig.canvas.manager.set_window_title(project_name)
        # Save the figure
        str_save_off = os.path.abspath(graphs_folder + project_name) + "/" + project_name + "-" + "assignees"

        for j, row in data.iterrows():
            # Iterate through all the data in streaming fashion
            # Gather together issues whose date of creation are within a time window of 30 days
            if row["created_date"] <= date + window_days:
                # For each unique assignee find the number of issues that were assigned to him within this time window
                assignee_count[row["assignee"]] += 1
                # print(assignee_count)
            else:
                # Plot the data
                for i in range(0, uniq_assignees):
                    axes[i].clear()
                    # A 2D list, where each element is a list containing the number of assigned issues
                    # to a unique assignee within the aforementioned time window throughout the whole dataset
                    results[i].append(assignee_count[i])
                    axes[i].plot(results[i], "red")
                    axes[i].set_facecolor(soft_blue)  # Soft blue color
                    axes[i].grid(True, color='white')
                    axes[i].set_title(f"Assignee {i}")
                    axes[i].set_ylabel("Issues")
                    axes[i].set_xlabel("Months")

                # Increase the time window variable to replicate the next month
                date += window_days
                # Clear the list when the time window has passed
                assignee_count = [0] * uniq_assignees
                # Don't forget to process the current observation as it doesn't belong to the previous time window
                assignee_count[row["assignee"]] += 1

        plt.tight_layout()
        # plt.show()
        fig.savefig(str_save_off)

print("--- %s seconds ---" % (time.time() - start_time))