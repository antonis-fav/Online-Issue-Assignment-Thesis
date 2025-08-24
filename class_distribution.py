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



        uniq_assignees = len(data["assignee"].unique())
        assignee_count = [0] * uniq_assignees


        soft_blue = (0.9, 0.95, 1)  # Light blue color

        x_axis=[]

        # Create a separate folder for each project's graphs
        if not os.path.exists(os.path.join(graphs_folder, "0_class_distribution")):
            os.makedirs(graphs_folder + "0_class_distribution")

        # Save the figure
        str_save_off = os.path.abspath(graphs_folder + "0_class_distribution") + "/" + project_name + "-" + "class_distribution"

        for j, row in data.iterrows():

            assignee_count[row["assignee"]] += 1

        for i in range(0, len(assignee_count)):
            x_axis.append(i)

        if len(assignee_count) <=5 :
            bar_width = 0.3
        elif len(assignee_count) <= 10:
            bar_width = 0.25
        elif len(assignee_count) <= 15:
            bar_width = 0.20
        elif len(assignee_count) < 20:
            bar_width = 0.20
        else:
            bar_width = 0.15

        plt.figure(figsize=(16,10))
        bars = plt.bar(x_axis, assignee_count, color="skyblue", width = bar_width)

        plt.xlabel('Classes', fontsize=18, fontweight='bold', labelpad=15)
        plt.ylabel('Number of Instances', fontsize=18, fontweight='bold', labelpad=15)

        title = "Class Distribution in " + project_name
        plt.title(title, fontsize=18, fontweight='bold', pad=15)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{yval}', ha='center', fontsize=10,
                     fontweight='bold')

        plt.xticks(ticks=x_axis, fontsize=10, fontweight='bold')
        plt.tight_layout()


        plt.savefig(str_save_off)

        # plt.show()



print("--- %s seconds ---" % (time.time() - start_time))
