import pandas as pd
import matplotlib.pyplot as plt
from river import feature_extraction, compose, metrics, tree, drift, preprocessing, drift, neighbors, tree, linear_model, naive_bayes, ensemble, multiclass
from sympy.abc import alpha
import numpy as np
from properties import data_folder, graphs_folder
import os
import time
import copy

start_time = time.time()

# A flag that indicates whether the model should be reset upon finding a drift.
reset_flag = 0

avg_acc_all_projects = []

# Create the figure and axis
fig, ax = plt.subplots(figsize=(24, 16))

for counter in range(1, 7, 1):
	# A list that holds the accuracy result for each project
	metric_table = []
	# Values for the X axis of the plot
	x_axis = []


	for f in os.listdir(data_folder):

		if f.startswith("2_"):

			my_model = ensemble.AdaBoostClassifier(
				model=(
					multiclass.OneVsRestClassifier(naive_bayes.MultinomialNB(alpha=0.1))
				),
				n_models=10,
				seed=42
			)

			if counter == 1:
				transformer = compose.TransformerUnion(
					("summary", feature_extraction.TFIDF(on="summary"))
				)

			elif counter == 2:
				transformer = compose.TransformerUnion(
					("summary", feature_extraction.TFIDF(on="summary")),
					("description", feature_extraction.TFIDF(on="description"))
				)
			elif counter == 3:
				transformer = compose.TransformerUnion(
					("summary", feature_extraction.TFIDF(on="summary")),
					("description", feature_extraction.TFIDF(on="description")),
					("labels", feature_extraction.TFIDF(on="labels"))
				)
			elif counter == 4:
				transformer = compose.TransformerUnion(
					("summary", feature_extraction.TFIDF(on="summary")),
					("description", feature_extraction.TFIDF(on="description")),
					("labels", feature_extraction.TFIDF(on="labels")),
					("components_name", feature_extraction.TFIDF(on="components_name"))
				)

			elif counter == 5:
				transformer = compose.TransformerUnion(
					("summary", feature_extraction.TFIDF(on="summary")),
					("description", feature_extraction.TFIDF(on="description")),
					("labels", feature_extraction.TFIDF(on="labels")),
					("components_name", feature_extraction.TFIDF(on="components_name")),
					("priority_name", feature_extraction.TFIDF(on="priority_name"))
				)

			elif counter == 6:
				transformer = compose.TransformerUnion(
					("summary", feature_extraction.TFIDF(on="summary")),
					("description", feature_extraction.TFIDF(on="description")),
					("labels", feature_extraction.TFIDF(on="labels")),
					("components_name", feature_extraction.TFIDF(on="components_name")),
					("priority_name", feature_extraction.TFIDF(on="priority_name")),
					("issue_type_name", feature_extraction.TFIDF(on="issue_type_name")),
				)

			### Reset the model for each project
			model = copy.deepcopy(my_model)

			# The implementation of the concept drift was acquired directly from the following example here:
			# https://riverml.xyz/latest/introduction/getting-started/concept-drift-detection/
			drift_detector = drift.ADWIN(delta=0.02)
			drifts = []

			# Read data for a project
			data = pd.read_csv(os.path.join(data_folder, f))
			project_name = data["project_name"].iloc[0]
			print("Processing : ", project_name)

			x_axis.append(project_name)

			# Create a separate folder for each project's graphs
			if not os.path.exists(os.path.join(graphs_folder, project_name)):
				os.makedirs(graphs_folder + project_name)

			# Reverse the order of the rows in the dataframe
			data = data.iloc[::-1]
			data = data.reset_index()

			# Convert the created_date variable of every issue into it's toordinal form
			data["created_date"] = pd.to_datetime(data["created_date"])
			data["created_date"] = data["created_date"].map(lambda x: x.toordinal())

			data = data.sort_values(by='created_date')

			# Use a metric for evaluating online the model
			metric = metrics.Accuracy()
			results = []

			for i, row in data.iterrows():

				# Iterate row wise through the dataset
				x, y = row, row["assignee"]


				# Remove all the keys/value pairs that won't be features in our model
				for key in ["index", "id", "project_name", "created_date", "assignee"]:
					x.pop(key)

				transformer.learn_one(x)
				x = transformer.transform_one(x)

				# Apply the model (predict the next instance and then train the model on it)
				y_pred = model.predict_one(x)
				model.learn_one(x, y)


				# Update the selected metric values
				metric.update(y, y_pred)
				result = metric.get()
				results.append(result)
				# Scan for drifts
				drift_detector.update(result)

				if drift_detector.drift_detected:
					# print(f"Change detected at index {i}")
					drifts.append(i)
					if reset_flag:
						model = copy.deepcopy(my_model)
			print(metric)
			metric_table.append(metric.get())



	print(metric_table)
	print("Average Accuracy across all projects: ", sum(metric_table) / len(metric_table))
	avg_acc_all_projects.append(sum(metric_table) / len(metric_table))


	plt.rcParams.update({'font.size': 24})  # You can adjust the size here

	# Number of bars (12 in this case)
	n = len(metric_table)

	# X-axis values
	x = np.arange(n)

	width = 0.15

	avg = sum(metric_table) / len(metric_table)

	if counter == 1:
		ax.axhline(y=avg, color='purple', linestyle='--', linewidth=2, zorder=1)
		bars1 = ax.bar(x - 2.5 * width, metric_table, width, color="purple", label='Title', zorder=2)
	if counter == 2:
		ax.axhline(y=avg, color='orange', linestyle='--', linewidth=2, zorder=1)
		bars2 = ax.bar(x - 1.5 * width, metric_table, width, color="orange", label='Title + Description', zorder=2)
	if counter == 3:
		ax.axhline(y=avg, color='blue', linestyle='--', linewidth=2, zorder=1)
		bars1 = ax.bar(x - 0.5 * width, metric_table, width, color="blue", label='Title + Description + Labels', zorder=2)
	if counter == 4:
		ax.axhline(y=avg, color='yellow', linestyle='--', linewidth=2, zorder=1)
		bars1 = ax.bar(x + 0.5 * width, metric_table, width, color="yellow", label='Title + Description + Components', zorder=2)
	if counter == 5:
		ax.axhline(y=avg, color='black', linestyle='--', linewidth=2, zorder=1)
		bars1 = ax.bar(x + 1.5 * width, metric_table, width, color="black", label='Title + Description + Components + Priority', zorder=2)
	if counter == 6:
		ax.axhline(y=avg, color='red', linestyle='--', linewidth=2, zorder=1)
		bars1 = ax.bar(x + 2.5 * width, metric_table, width, color="red", label='Title + Description + Components + Priority + Issue_Type', zorder=2)

	# Adding labels and title
	ax.set_xlabel('Project', fontsize=24, fontweight='bold')
	ax.set_ylabel('Moving Average Accuracy', fontsize=24, fontweight='bold')
	# Set y-ticks (to make the labels more readable)
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	# Customize font size or tick size to make them stand out more
	ax.tick_params(axis='y', labelsize=12)  # Increases size of y-tick labels
	ax.set_title("Accuracy Performance among different Configurations " , fontsize=24, fontweight='bold')
	ax.set_xticks(x)
	# Set x-axis label size and rotate the labels for better readability
	ax.tick_params(axis='x', labelsize=12)  # Increases size of x-tick labels

	ax.set_xticklabels(i for i in x_axis)

	plt.setp(ax.get_yticklabels(), fontweight='bold')
	plt.setp(ax.get_xticklabels(), fontweight='bold')

	# Manually add a single legend entry for the average line
	avg_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1)

	# Combine all handles for the legend (bars and the average line)
	handles, labels = ax.get_legend_handles_labels()
	handles.append(avg_line)
	labels.append('Avg value')

	# Set the legend with all required labels
	ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=24)

	plt.tight_layout()

plt.savefig(os.path.abspath(graphs_folder) + "/" + "5.3_Contribution_Different_Configurations.png")
plt.show()

print("Average accuracy accross all projects for each configuration is: ", avg_acc_all_projects)

print("--- %s seconds ---" % (time.time() - start_time))
