#!/usr/bin/env python
# Author: Brandon M Booth

import os
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))

from util import GetLocalTimestampFromUnixTime
from util import GetUnixTimeFromTimestamp

def PreprocessData(fitbit_data_path, output_path):
   if not os.path.exists(output_path):
      os.makedirs(output_path)

   all_fitbit_files = glob.glob(os.path.join(fitbit_data_path, '*.csv.gz'))
   daily_summary_files = [x for x in all_fitbit_files if 'dailySummary' in x]
   for daily_summary_file in daily_summary_files:
      pid = os.path.basename(daily_summary_file).split('_')[0]

      #if not '0a85fd46' in daily_summary_file:
      #   continue

      # Data to time slice
      hr_file_path = os.path.join(os.path.dirname(daily_summary_file),pid+'_heartRate.csv.gz')
      steps_file_path = os.path.join(os.path.dirname(daily_summary_file),pid+'_stepCount.csv.gz')
      hr_df = pd.read_csv(hr_file_path)
      steps_df = pd.read_csv(steps_file_path)

      # Time slice all data according to the awake periods in between the main sleep periods
      daily_df = pd.read_csv(daily_summary_file)
      daily_df_main_sleep = daily_df.loc[:,['Sleep1BeginTimestamp','Sleep1EndTimestamp']]
      daily_df_main_sleep.append([np.nan, np.nan])# Add dummy row at the end
      sleep_end_timestamp = np.nan
      for row_idx in range(daily_df_main_sleep.shape[0]):
         sleep_start_timestamp = daily_df_main_sleep['Sleep1BeginTimestamp'].iloc[row_idx]
         output_file_path_prefix = os.path.join(output_path, pid)

         # Slice Fitbit HR
         hr_mask = None
         if not pd.isna(sleep_start_timestamp):
            hr_mask = hr_df['Timestamp'] < sleep_start_timestamp
         if not pd.isna(sleep_end_timestamp):
            if hr_mask is not None:
               hr_mask = np.logical_and(hr_mask, hr_df['Timestamp'] > sleep_end_timestamp)
            else:
               hr_mask = hr_df['Timestamp'] > sleep_end_timestamp
         if hr_mask is not None:
            sliced_hr_df = hr_df.loc[hr_mask, :]
            sliced_hr_df.to_csv(output_file_path_prefix+'_heartRate_day'+str(row_idx)+'.csv', index=False, header=True)

         # Slice Fitbit step count
         steps_mask = None
         if not pd.isna(sleep_start_timestamp):
            steps_mask = steps_df['Timestamp'] < sleep_start_timestamp
         if not pd.isna(sleep_end_timestamp):
            if steps_mask is not None:
               steps_mask = np.logical_and(steps_mask, steps_df['Timestamp'] > sleep_end_timestamp)
            else:
               steps_mask = steps_df['Timestamp'] > sleep_end_timestamp
         if steps_mask is not None:
            sliced_steps_df = steps_df.loc[steps_mask, :]
            sliced_steps_df.to_csv(output_file_path_prefix+'_stepCount_day'+str(row_idx)+'.csv', index=False, header=True)

         sleep_end_timestamp = daily_df_main_sleep['Sleep1EndTimestamp'].iloc[row_idx]

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-f', '--fitbit_preprocessed_data_path', required=True, help='Path to folder containing TILES preprocessed fitbit data')
   parser.add_argument('-o', '--output_path', required=True, help='Output folder for preprocessed data')
   try:
      args = parser.parse_args()
   except SystemExit as e:
      if e.code == 2:
         parser.print_help()
         sys.exit(0)

   PreprocessData(args.fitbit_preprocessed_data_path, args.output_path)
