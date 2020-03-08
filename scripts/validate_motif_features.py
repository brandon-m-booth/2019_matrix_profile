import os
import sys
import pdb
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))
from util import GetUnixTimeFromTimestamp

def RunValidateMotifFeatures(tiles_data_root_path, motif_feats_path, output_path):
   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   # Load motif features and obtain list of participant IDs to test
   motif_files = glob.glob(os.path.join(motif_feats_path, '*.csv'))
   pids = []
   motif_dfs = {}
   for motif_file in motif_files:
      pid = os.path.basename(motif_file).split('_')[0]
      pids.append(pid)
      pid_df = pd.read_csv(motif_file)
      # HACK!!  - Add a timestamp colum to the motif feature file
      heart_rate_path = os.path.join(motif_feats_path, '..', 'force_aligned3', 'fitbit')
      hr_files = glob.glob(os.path.join(heart_rate_path, '*.csv.gz'))
      pid_hr_file = [x for x in hr_files if os.path.basename(x).startswith(pid)][0] # There should be only one
      pid_hr_df = pd.read_csv(pid_hr_file)
      ts_col = pid_hr_df['Timestamp']
      ts_unix_col = GetUnixTimeFromTimestamp(ts_col.tolist())
      pid_df['UnixTime'] = ts_unix_col
      # END HACK!!!
      motif_dfs[pid] = pid_df

   # TODO - Load other features
   feats_dfs = motif_dfs

   # Load MGT self-reported constructs
   mgt_files = glob.glob(os.path.join(tiles_data_root_path, 'surveys/scored/EMAs/*.csv.gz'))

   # TODO - Include more than just anxiety
   # Load the appropriate MGT survey columns for each mgt file
   pid_col = 'participant_id'
   start_ts_col = 'start_ts'
   end_ts_col = 'completed_ts'
   always_load_cols = [pid_col, start_ts_col, end_ts_col]
   mgt_cols_map = {'anxiety.csv.gz': ['anxiety']}

   mgt_dfs = {}
   for mgt_file in mgt_files:
      file_name = os.path.basename(mgt_file)
      if file_name in mgt_cols_map.keys():
         cols_to_load = always_load_cols+mgt_cols_map[file_name]
         df = pd.read_csv(mgt_file)
         filtered_df = df.loc[:,cols_to_load]
         mgt_dfs[file_name] = filtered_df

   # Validate the motif features
   # TODO - Grid search window size and rf_depth
   max_time_window_sec = 86400 # One day
   rf_depth = 10
   for pid in pids:
      pid_feats_df = feats_dfs[pid]

      mgt_pred_scores = {}
      for mgt_df_item in mgt_dfs.items():
         mgt_filename = mgt_df_item[0]
         mgt_df = mgt_df_item[1]
         pid_mgt_df = mgt_df.loc[mgt_df.loc[:,pid_col] == pid,:]
         start_time_unix = GetUnixTimeFromTimestamp(pid_mgt_df[start_ts_col], parse_expr='%Y-%m-%dT%H:%M:%S')
         end_time_unix = GetUnixTimeFromTimestamp(pid_mgt_df[end_ts_col], parse_expr='%Y-%m-%dT%H:%M:%S')
         seconds_since_last_same_survey = np.diff(start_time_unix).tolist()
         seconds_since_last_same_survey.insert(0, np.inf)
         window_seconds = np.minimum(seconds_since_last_same_survey, max_time_window_sec)
         window_end_time_unix = start_time_unix
         window_start_time_unix = window_end_time_unix - window_seconds

         label_cols = mgt_cols_map[mgt_filename]
         for label_col in label_cols:
            pid_mgt_label = pid_mgt_df[label_col]
            functionals = ['Mean', 'Median', 'Min', 'Max', 'Q1', 'Q3', 'Range', 'Var', 'Std',]
            pid_feats_cols = pid_feats_df.columns.tolist()
            pid_feats_cols.remove('UnixTime')
            data_cols = []
            for pid_feats_col in pid_feats_cols:
               for functional in functionals:
                  data_cols.append(pid_feats_col+functional)
            pid_data_df = pd.DataFrame(data=np.zeros((len(pid_mgt_label), len(data_cols))), columns=data_cols)
            for i in range(len(pid_mgt_label)):
               start_unix = window_start_time_unix[i]
               end_unix = window_end_time_unix[i]
               window_mask = np.logical_and(pid_feats_df['UnixTime'] >= start_unix, pid_feats_df['UnixTime'] <= end_unix)
               pid_feats_window_df = pid_feats_df.loc[window_mask, :]
               pid_feats_window_df = pid_feats_window_df.drop(['UnixTime'], axis=1)

               # Compute functionals of features over the window
               for pid_feat_col in pid_feats_window_df.columns:
                  pid_data_df.loc[i, pid_feat_col+'Mean'] = pid_feats_window_df[pid_feat_col].mean(axis=0)
                  pid_data_df.loc[i, pid_feat_col+'Median'] = pid_feats_window_df[pid_feat_col].median(axis=0)
                  pid_data_df.loc[i, pid_feat_col+'Min'] = pid_feats_window_df[pid_feat_col].min(axis=0)
                  pid_data_df.loc[i, pid_feat_col+'Max'] = pid_feats_window_df[pid_feat_col].max(axis=0)
                  pid_data_df.loc[i, pid_feat_col+'Q1'] = pid_feats_window_df[pid_feat_col].quantile(q=0.25)
                  pid_data_df.loc[i, pid_feat_col+'Q3'] = pid_feats_window_df[pid_feat_col].quantile(q=0.75)
                  pid_data_df.loc[i, pid_feat_col+'Range'] = pid_data_df.loc[i, pid_feat_col+'Max'] - pid_data_df.loc[i, pid_feat_col+'Min']
                  pid_data_df.loc[i, pid_feat_col+'Var'] = pid_feats_window_df[pid_feat_col].var(axis=0)
                  pid_data_df.loc[i, pid_feat_col+'Std'] = pid_feats_window_df[pid_feat_col].std(axis=0)

            # Final data cleaning
            data_nan_mask = np.sum(np.isnan(pid_data_df), axis=1).astype(bool).tolist()
            label_nan_mask = np.isnan(pid_mgt_label).tolist()
            nan_mask = ~np.logical_or(data_nan_mask, label_nan_mask)
            pid_data_df = pid_data_df.loc[nan_mask,:]
            pid_mgt_label = pid_mgt_label[nan_mask]

            # Validation
            model = RandomForestClassifier(max_depth=rf_depth)
            cv_results = cross_validate(model, pid_data_df, pid_mgt_label)
            score_mean = np.mean(cv_results['test_score'])
            score_std = np.std(cv_results['test_score'])
            mgt_pred_scores[label_col] = (score_mean, score_std)

      # Plot scores for participant
      mgt_pred_score_names = mgt_pred_scores.keys()
      mgt_pred_score_means, mgt_pred_score_std = zip(*mgt_pred_scores.values())
      x_pos = [i for i, _ in enumerate(mgt_pred_score_names)]
      plt.bar(x_pos, mgt_pred_score_means, yerr=mgt_pred_score_std)
      plt.xlabel(mgt_pred_score_names)
      plt.ylabel('CV Prediction Accuracy')
      plt.title('MGT RF Prediction Scores for Participant %s'%(pid))
      plt.xticks(x_pos, mgt_pred_score_names)
      plt.show()

   return


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--tiles_data_root_path', required=True, help='Path to the root folder of the published TILES data')
   parser.add_argument('--motif_path', required=True, help='Path to the folder containing motif features from find_motifs.py')
   parser.add_argument('--output_path', required=True, help='Output CSV file path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   RunValidateMotifFeatures(args.tiles_data_root_path, args.motif_path, args.output_path)
