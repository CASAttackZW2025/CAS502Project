import unittest
import pandas as pd
import json
import numpy as np

#Create testing data for 'ProcessorID', 'TaskName', and 'StationID' to test
#functionality of the DataSchema script.

task_data = {"ProcessorID": [101, 102, 103, 104],
    "TaskName": ["Task A", "Task B", "Task C", "Task D"]}

station_data = {"ProcessorID": [101, 102, 103],
    "StationID": ["S1", "S2", "S3"]}

TaskProc_df = pd.DataFrame(task_data)
Station_df = pd.DataFrame(station_data)

ProcStat_dict = pd.Series(Station_df["StationID"].values, index=Station_df["ProcessorID"]).to_dict()

def Sort_Stations(df, dictionary, column_name: str, key_column: str):
    df[column_name] = df[key_column].map(dictionary)
    df[column_name] = df[column_name].where(pd.notna(df[column_name]), None)
    return df

class TestSortStations(unittest.TestCase):

#Test to determine if the function correctly maps stations to processors and tasks.
#This test will not only test the function but also the effectiveness of the dictionary

    def test_correct_mapping(self):
        df_result = Sort_Stations(TaskProc_df.copy(), ProcStat_dict, "StationID", "ProcessorID")
        expected_stations = ["S1", "S2", "S3", None]  # ProcessorID 104 is missing in the dictionary
        self.assertListEqual(df_result["StationID"].tolist(), expected_stations)

#Test to see if the function will be able to identify when a processor does not have a linked station.

    def test_missing_processor_id(self):
        df_result = Sort_Stations(TaskProc_df.copy(), ProcStat_dict, "StationID", "ProcessorID")
        self.assertIsNone(df_result.loc[df_result["ProcessorID"] == 104, "StationID"].values[0])

#Test if the function will break when fed blank information for ProcessorID and TaskName.
#The function should recognize it is blank and not produce an error.
 
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["ProcessorID", "TaskName"])
        df_result = Sort_Stations(empty_df, ProcStat_dict, "StationID", "ProcessorID")
        self.assertTrue(df_result.empty)

