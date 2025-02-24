# CAS502Project
Cross Training Queuing System

## Team Members: Zack Weber & Guy McFall

## Project Description:
This Python project will develop a workforce allocation and production scheduling system designed to simulate and optimize resource management in manufacturing environments. It will ideally use a flexible framework to dynamically assign, simulate, and optimize worker assignments across multiple processors, stations, and shifts. The primary objective is to load, simulate, and save the simulation history, results, and statistics for production systems with workers and resource constraints. Our stretch goals are to employ a mathematical model, where the software will calculate the most efficient worker configurations/combinations, minimize idle time and ensure production targets are met without having to run the simulation for each iteration to search through each configuration combination directly.

## User Instructions:
There a couple of key notes to be able to use this software correctly. First, you must use the intake form provided. Read through the "Intake Form Example" to see what a filled out version looks like. Second, in order to properly run our script you must call open it in google colab. The script will prompt you to upload your intake form. From there it will run the simulations and produce the idle times associated with your tasks, processors, stations, and resources.
User Instructions:
There a couple of key notes to be able to use this software correctly. First, you must use the intake form provided. Read through the intake Form Example to see what a filled out version looks like. Second, in order to properly run our script you must call open it in google colab. The script will prompt you to upload your intake form. From there it will run the simulations and produce the idle times associated with your tasks, processors, stations, and resources.

### Fill out the Intake Form-
#### Download & Review the Intake Form:
First, you must use the intake form provided. 
Locate the provided intake form (e.g., production_system_intake_Example01.xlsx) in the repository.
Read through the "Intake Form Example" to see what a filled out version looks like.

#### Read the Detailed Guidelines:
Open the accompanying Intake form Instructions.md/.docx file for step-by-step item by item details for filling out the form. This file explains what each data field is and what is required for each section and production_system_intake_Example01.xlsx provides examples of a correctly completed form.

#### Fill Out Your Form:
Complete every required field in each sheet (e.g., ProcessorTypeTbl, LocationTbl, TaskTbl, etc.) with accurate information that reflects your production system.

### Run the Simulation in Google Colab-
#### Open the Colab Notebook (for now it is T):
Click the "Open in Google Colab" button from the repository and sign in with your Google account if prompted.

#### Upload Your Intake Form:
When you run the ingestion cell, the script will display a file upload widget. Click "Choose File" and select your completed intake form.

#### Execute the Simulation:
The script will:

Use the ingest_excel_to_dfs() function to load each sheet into a Pandas DataFrame.
Call create_table_mapping() to generate a summary of each table (including row and column counts).
Run the simulation, process your data, and produce results (e.g., idle time metrics, worker assignments, etc.).
#### Review the Output:
Check the "TABLE MAPPING SUMMARY" and review the detailed descriptive statistics for each table. This output helps verify that your form was read correctly and shows the simulation results.

## Reporting Bugs:
To report bugs identified in the code, please leverage the "issues" function on github. You should navigate to the main page on the repository and click on the "issues" and "new issue" tabs.

## Collaboration:
If you would like to collaborate with this repository you may follow one of two approaches. In most instances, collaborators will not need direct access and can thus fork the repository and clone your fork. Once you make changes in your cloned repository, please come to this repository and create a pull request by clicking "compare & pull request." Be sure to write a description of the changes you propose.

If you feel that you need direct access, you may create an issue in the repository requesting access.

## Project Details-
### Challenges:

### Technical:
-Creating a consistent data structure for loading and saving the details of the production system can be complicated because these production systems can have lots of complex features, interactions, and requirements.
-One challenge is that data for this type of model does not come easy. We have several real-world examples but the data is perhaps not generic enough for this tool. We will likely need to develop “dummy” data to develop and test this software.
-Optimization models can be challenging, will we be able to develop the correct script to produce an output that genuinely maximizes the output while minimizing resources and idle time ratio?

### Process:
-Working in a group of two on a complicated project with a short timeline will require clear and open communication;
-Roles and tasks must be defined and understood by both parties; and
-R&D projects have constant changes as new information is received. We will need to be agile and able to pivot quickly considering the short timeline.

### Team and Skills:
-We will both need to understand each-other’s strengths and divvy up tasks that align with those strengths. However, we will both need to share the development tasks regardless of varying levels of technical skillset. This will require strong communication, trust, and dependability.
Work

### Work & Communication Plan:

We each bring different strengths to the project. Zack has experience in manufacturing and is familiar with the considerations and theory behind optimizing resource management. Guy is a consultant manager with experience in developing optimization models and general project management. Pairing these skill sets together will drive the team towards successfully developing a minimum viable product.

-Weekly Meetings- We will have a standing dedicated Teams meeting once a week to discuss live progress, pending tasks, challenges, and any potential pivots.

-Backlog Management- We will employ a Teams planner to track our backlog that defines various tasks, who is responsible for what, and when are items due.

### Branching Methodology:
We will follow a "Feature Branches Workflow" methodology. Our main branch will be for the primary features. We will have branches for data intake, data ingestion, data schema, output plots, output tables, and output statistics. We will split up the various branches for who is taking lead. When a feature is done and ready to be incorporated with the main branch, the non-lead partner will perform a peer review. Once accomplished, we will merge that branch with the main.
