#!/usr/bin/env python3
# Author: Brandon M Booth

import os
import sys
import pdb
import glob
import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))

from util import GetLocalTimestampFromUnixTime
from util import GetUnixTimeFromTimestamp

def ForceAlignDataFrame(df, hr_df):
   hr_vals = hr_df['HeartRatePPG']
   hr_timestamps = hr_df['Timestamp']
   hr_interp = df['HeartRatePPG'].interpolate(method='linear').values
   df['HeartRatePPGOld'] = hr_interp

   # Don't allow interpolated HR values if the time between samples is too long
   hr_seconds_threshold = 20
   hr_invalid_val = 70 # 70 BPM
   hr_invalid_min_seq_len = 10
   hr_unix_times = GetUnixTimeFromTimestamp(hr_timestamps)
   interp_unix_time = GetUnixTimeFromTimestamp(df['Timestamp'])

   hr_idx = 0
   do_nan_fill = True
   for interp_idx in range(len(hr_interp)):
      if hr_idx < len(hr_unix_times) and interp_unix_time[interp_idx] >= hr_unix_times[hr_idx]:
         do_nan_fill = False
         if hr_idx+1 >= len(hr_vals):
            do_nan_fill = True
         elif (hr_unix_times[hr_idx+1] - hr_unix_times[hr_idx]) >= hr_seconds_threshold:
            do_nan_fill = True
         elif hr_vals[hr_idx+1] == hr_invalid_val and hr_vals[hr_idx] == hr_invalid_val:
            # Figure out how long the sequence of invalid values is. If long enough, nan fill
            start_invalid_hr_idx = hr_idx
            while start_invalid_hr_idx > 0:
               if hr_vals[start_invalid_hr_idx] != hr_invalid_val:
                  break
               start_invalid_hr_idx -= 1
            end_invalid_hr_idx = hr_idx
            while end_invalid_hr_idx < len(hr_unix_times):
               if hr_vals[end_invalid_hr_idx] != hr_invalid_val:
                  break
               end_invalid_hr_idx += 1
            if end_invalid_hr_idx - start_invalid_hr_idx >= hr_invalid_min_seq_len:
               do_nan_fill = True
         hr_idx += 1
      if do_nan_fill:
         hr_interp[interp_idx] = np.nan
   df['HeartRatePPG'] = hr_interp

   # Align to the StepCount data
   step_count_valid_mask = ~pd.isna(df['StepCount'])
   df = df.loc[step_count_valid_mask,:]

   return df

def PreprocessData(fitbit_data_path, output_path):
   if not os.path.exists(output_path):
      os.makedirs(output_path)

   all_fitbit_files = glob.glob(os.path.join(fitbit_data_path, '*.csv.gz'))
   pids = np.unique([os.path.basename(x).split('_')[0] for x in all_fitbit_files])
   daily_summary_files = [x for x in all_fitbit_files if 'dailySummary' in x]
   for pid in tqdm.tqdm(pids):
      pid_fitbit_files = [x for x in all_fitbit_files if pid in x]
      pid_daily_summary_files = [x for x in pid_fitbit_files if 'dailySummary' in x]
      pid_hr_files = [x for x in pid_fitbit_files if 'heartRate' in x]
      pid_steps_files = [x for x in pid_fitbit_files if 'stepCount' in x]

      pid_hr_dfs = []
      for pid_hr_file in pid_hr_files:
         pid_hr_dfs.append(pd.read_csv(pid_hr_file))
      pid_combined_hr_df = pd.concat(pid_hr_dfs)
      pid_combined_hr_df = pid_combined_hr_df.sort_values(by=['Timestamp'])
      pid_steps_dfs = []
      for pid_steps_file in pid_steps_files:
         pid_steps_dfs.append(pd.read_csv(pid_steps_file))
      pid_combined_steps_df = pd.concat(pid_steps_dfs)

      pid_combined_df = pid_combined_hr_df.merge(pid_combined_steps_df, how='outer')
      pid_combined_df = pid_combined_df.sort_values(by=['Timestamp'])
      pid_combined_df.index = range(pid_combined_df.shape[0])
      pid_combined_aligned_df = ForceAlignDataFrame(pid_combined_df, pid_combined_hr_df)

      # Add binary flag column according to the awake periods in between the main sleep periods
      pid_combined_aligned_df['AwakeFlag'] = 1
      for pid_daily_summary_file in pid_daily_summary_files:
         pid_daily_df = pd.read_csv(pid_daily_summary_file)
         for row_idx in range(pid_daily_df.shape[0]):
            sleep_start = pid_daily_df['Sleep1BeginTimestamp'].iloc[row_idx]
            sleep_end = pid_daily_df['Sleep1EndTimestamp'].iloc[row_idx]
            mask = pid_combined_aligned_df['Timestamp'] > sleep_start
            mask = np.logical_and(mask, pid_combined_aligned_df['Timestamp'] < sleep_end)
            pid_combined_aligned_df.loc[mask, 'AwakeFlag'] = 0

      #plt.plot(pid_combined_aligned_df.index, pid_combined_aligned_df['HeartRatePPGOld'], color='orange')
      #plt.plot(pid_combined_aligned_df.index, pid_combined_aligned_df['HeartRatePPG'], color='blue')
      #plt.show()
      pid_combined_aligned_df.to_csv(os.path.join(output_path, pid)+'_force_aligned.csv.gz', index=False, header=True, compression='gzip')
   return

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
