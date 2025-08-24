import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from river import feature_extraction, compose, metrics, tree, drift, preprocessing, drift, neighbors, tree, linear_model, naive_bayes, ensemble, multiclass
from properties import data_folder, graphs_folder

# DESC: Create a batch learning model based on Naive Bayes Multinomial, that will be compared with the Online Learning Model

start_time = time.time()

# A flag that indicates whether the model should be reset upon finding a drift.
reset_flag = 1
# A list that holds the accuracy result for each project
metric_table = []
# A list that holds the average accuracy of Online Learning Model on the given test set for each project
acc_online_test = []

max_corpus_labels = 1000
max_corpus_nlp = 2000
min_df_labels = 5
min_df_nlp = 0.0

# A list that holds the average accuracy of Batch Learning Model on the given test set for each project
acc_batch_test = []
rec_all = []
f1_all = []

x_axis = []

plt.rcParams.update({'font.size': 24}, )  # You can adjust the size here


# Convert every feature represented in natural language format into a tfidf vector
def tfidf_fun(training_data, testing_data, max_corpus, min_docfreq):

    # The max words of the vocabulary will equal to max_features
    tfidf_vectorizer = TfidfVectorizer(max_features=max_corpus, min_df=min_docfreq)
    # Some issues had np.nan values inside the fields summary,description or labels, we handle them with the astype('U')
    X_train_vec = tfidf_vectorizer.fit_transform(training_data.values.astype('U'))
    # We only preform transformation on  the test data, we don't perform fit on them
    X_test_vec = tfidf_vectorizer.transform(testing_data.values.astype('U'))

    return X_train_vec, X_test_vec


for f in os.listdir(data_folder):
    if f.startswith("2_"):

        # Read the data for each project
        data = pd.read_csv(os.path.join(data_folder, f))

        # Reverse the order of the rows in the dataframe
        data = data.iloc[::-1]
        data = data.reset_index()

        # Convert the created_date variable of every issue into it's toordinal form
        data["created_date"] = pd.to_datetime(data["created_date"])
        data["created_date"] = data["created_date"].map(lambda x: x.toordinal())
        # Sort the data by ascending chronological order
        data = data.sort_values(by='created_date')

        # Number of assignees working on the current project
        n_assignees = data["assignee"].unique()

        # Our target class variable will be the assignee id
        y_train = data["assignee"]

        project_name = data["project_name"].iloc[0]

        print("processing: ", project_name)

        # Create a separate folder for this specific experiment of section 5.1
        if not os.path.exists(os.path.join(graphs_folder)):
            os.makedirs(graphs_folder)

        # Maintain only the necessary features
        X_train = data.drop(["id", "project_name", "assignee", "created_date"], axis=1)



        # Create the train validation (used later) and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

        X_train_sum, X_test_sum = tfidf_fun(X_train["summary"], X_test["summary"], max_corpus_nlp, min_df_nlp)
        X_train_desc, X_test_desc = tfidf_fun(X_train["description"], X_test["description"], max_corpus_nlp, min_df_nlp)
        X_train_labels, X_test_labels = tfidf_fun(X_train["labels"], X_test["labels"], max_corpus_labels, min_df_labels)
        X_train_components, X_test_components = tfidf_fun(X_train["components_name"], X_test["components_name"], max_corpus_labels, min_df_labels)
        X_train_priority, X_test_priority = tfidf_fun(X_train["priority_name"], X_test["priority_name"], max_corpus_labels, min_df_labels)
        X_train_issue_type, X_test_issue_type = tfidf_fun(X_train["issue_type_name"], X_test["issue_type_name"], max_corpus_labels, min_df_labels)



        X_train_combined = hstack([X_train_sum, X_train_desc, X_train_labels, X_train_components, X_train_priority, X_train_issue_type])

        X_test_combined = hstack([X_test_sum, X_test_desc, X_test_labels, X_test_components, X_test_priority, X_test_issue_type])

        model = MultinomialNB(alpha = 0.1)

        # Train a KNN model for every feature separately
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_test_combined)

        acc = accuracy_score(y_pred, y_test, normalize=True)

        print("Accuracy of Batch Learning Model on test set is: ", acc)
        acc_batch_test.append(acc)

print("The average accuracy of Batch Learning Model across all projects is: ", sum(acc_batch_test)/len(acc_batch_test))




### Online Learning ###

for f in os.listdir(data_folder):
    if f.startswith("2_"):

        # The implementation of the concept drift was acquired directly from the following example here:
        # https://riverml.xyz/latest/introduction/getting-started/concept-drift-detection/
        drift_detector = drift.ADWIN(delta=0.00001)
        drifts = []

        # Read data for a project
        data = pd.read_csv(os.path.join(data_folder, f))
        project_name = data["project_name"].iloc[0]
        print("Processing : ", project_name)

        x_axis.append(project_name)

        data = data.iloc[::-1]
        data = data.reset_index()

        # Convert the created_date variable of every issue into it's toordinal form
        data["created_date"] = pd.to_datetime(data["created_date"])
        data["created_date"] = data["created_date"].map(lambda x: x.toordinal())
        # Sort the data by ascending chronological order
        data = data.sort_values(by='created_date')

        y_train = data["assignee"]
        # Training data will contain "summary", "description", "labels", "components"
        X_train = data.drop(["id", "project_name","assignee", "created_date"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42,shuffle=False)

        transformer = compose.TransformerUnion(
            ("summary", feature_extraction.TFIDF(on="summary")),
            ("description", feature_extraction.TFIDF(on="description")),
            ("labels", feature_extraction.TFIDF(on="labels")),
            ("components_name", feature_extraction.TFIDF(on="components_name")),
            ("priority_name", feature_extraction.TFIDF(on="priority_name")),
            ("issue_type_name", feature_extraction.TFIDF(on="issue_type_name")),

        )
        my_model = naive_bayes.MultinomialNB(alpha=0.1)

        # Create a model that can detect drift
        import copy
        model = copy.deepcopy(my_model)

        # Use a metric for evaluating online the model
        metric = metrics.Accuracy()
        results = []

        # A counter that hold the number of correct classified instances of Online Learning Model on test set
        acc_count = 0

        for i, row in data.iterrows():

            # Iterate row wise through the dataset
            x, y = row, row["assignee"]

            # Remove all the keys/value pairs that won't be features in our model
            for key in ["index", "id", "project_name", "assignee"]:
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
                    # print("reset the models")
                    model = copy.deepcopy(my_model)

            if i > (len(y_train) - 1):
                if y_pred == y :
                    acc_count += 1


        acc_test = acc_count / len(y_test)
        acc_online_test.append(acc_test)
        print("accuracy of online model for test set is: ", acc_test)

        print("moving average accuracy of online model is: ", metric)
        metric_table.append(metric.get())
        # print(metric_table)




print("Average Accuracy across all projects of online model on test set is: ", sum(acc_online_test) / len(acc_online_test))
print("Average Accuracy across all projects: ", sum(metric_table) / len(metric_table))

# Plot the data for the Batch Learning and for the Online Learning Model on grouped bar charts

# Number of bars (12 in this case)
n = len(acc_online_test)

# X-axis values
x = np.arange(n)

# Adding an average horizontal lines for both Batch and Online Learning
batch_avg = np.mean(acc_batch_test)
online_avg = np.mean(acc_online_test)

# Width of the bars
width = 0.35

# Create the figure and axis
fig, ax = plt.subplots(figsize=(24, 16))

# Plot an average horizontal lines for both Batch and Online Learning
ax.axhline(y=batch_avg, color='blue', linestyle='--', linewidth=2, zorder=1)
ax.axhline(y=online_avg, color='orange', linestyle='--', linewidth=2, zorder=1)
# Plotting the bars for both lists
bars1 = ax.bar(x - width/2, acc_batch_test, width, label='Batch Learning', zorder=2)
bars2 = ax.bar(x + width/2, acc_online_test, width, label='Online Learning', zorder=2)

# Adding labels and title
ax.set_xlabel('Project', fontsize=24, fontweight='bold')
ax.set_ylabel('Accuracy on Test Set', fontsize=24, fontweight='bold')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.tick_params(axis='y', labelsize=18)  # Increases size of y-tick labels
ax.set_title('Comparison of Batch and Online Learning Models', fontsize=24, fontweight='bold')
ax.set_xticks(x)
ax.tick_params(axis='x', labelsize=18)  # Increases size of x-tick labels
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.set_xticklabels(i for i in x_axis)

# Manually add a single legend entry for the average line
avg_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1)

# Combine all handles for the legend (bars and the average line)
handles, labels = ax.get_legend_handles_labels()
handles.append(avg_line)
labels.append('Avg value')

# Set the legend with all required labels
ax.legend(handles=handles, labels=labels, loc='upper left')

plt.savefig(os.path.abspath(graphs_folder) + "/" + "5.1_Exp_Batch_Online_Learning.png")
plt.tight_layout()
# Show the plot
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
