import os
import json
import codecs
import pymongo
import time
from properties import data_folder, mongo_URL, min_issues_assignee, graphs_folder

start_time = time.time()

if not os.path.exists(data_folder):
	os.makedirs(data_folder)
if not os.path.exists(graphs_folder):
	os.makedirs(graphs_folder)

# Connect to database
client = pymongo.MongoClient(mongo_URL)
db = client["jidata"]


def issue_filter(project_name, assignee):
	return {
				# # Only issues with summary, description, and labels are kept
				'projectname': project_name,
				'summary': {'$exists': True, '$not': {'$size': 0}},
				'description': {'$exists': True, '$not': {'$size': 0}},
				'labels': {'$exists': True, '$not': {'$size': 0}},
				'assignee': assignee
			}


def issue_filter_all(project_name):
	return {
				# # Only issues with summary, description, and labels are kept
				'projectname': project_name,
				'summary': {'$exists': True, '$not': {'$size': 0}},
				'description': {'$exists': True, '$not': {'$size': 0}},
				'labels': {'$exists': True, '$not': {'$size': 0}},
				'assignee': {'$exists': True, '$not': {'$size': 0}}
			}


def pipeline(projectname, minissuesperassignee):
	return [
		{'$match': issue_filter(projectname, {'$exists': True, '$not': {'$size': 0}})},
		# Gather and count the issues for every unique assignee inside the project
		{'$group': {'_id': '$assignee', 'count': {'$sum': 1}}},
		# if the number of the issues is equal or greater than the minimum issues per assignee keep these issues
		{'$match': {'$and': [
			{'count': {'$gte': minissuesperassignee}},
			# There is an assigned developer
			{'_id': {'$ne': None}}
		]}},
		# Sort the number of issues per assignee by descending order
		{'$sort': {'count': -1}}
	]


# Find all projects and don't return an id for every returned document (_id: 0)
for project in db["projects"].find(projection={"projectname": 1, "_id": 0}):
	projectname = project["projectname"]

	# For each project find the assignees of at least 80 issues and keep only these
	# An array that contains the names of the assignees with more than 80 issues in this specific project
	projectassignees = [assignee["_id"] for assignee in db["issues"].aggregate(pipeline(projectname, min_issues_assignee))]
	# If there are at least 5 such assignees
	if len(projectassignees) >= 5:
		print("Processing project " + projectname)
		# Download all the issues of these assignees for the project
		mongo_filter = issue_filter(projectname, {'$in': projectassignees})
		# Download all the issues of the project
		mongo_filter_all = issue_filter_all(projectname)

		mongo_projection = {"summary": 1, "description": 1, "labels": 1, "components": 1, "priority": 1, "issuetype": 1,
							"projectname": 1, "created": 1, "assignee": 1}

		issues = [issue for issue in db["issues"].find(filter=mongo_filter, projection=mongo_projection)]
		issues_all = [issue for issue in db["issues"].find(filter=mongo_filter_all, projection=mongo_projection)]
		with codecs.open(data_folder + "1_" + projectname + ".json", 'w', 'utf-8') as outfile:
			json.dump(issues, outfile, indent=3, default=str)
		with codecs.open(data_folder + "1_" + projectname + "_all" + ".json", 'w', 'utf-8') as outfile:
			json.dump(issues_all, outfile, indent=3, default=str)

print("--- %s seconds ---" % (time.time() - start_time))
