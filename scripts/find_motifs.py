#!/usr/bin/env python3
#Author: Brandon M Booth

import os
import sys
import pdb
import glob
import stumpy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix

do_show_plot = False

def LoadAlignedTILESData(aligned_data_root_path):
   fitbit_path = os.path.join(aligned_data_root_path, 'fitbit')

   fitbit_aligned_files = glob.glob(os.path.join(fitbit_path, '*_force_aligned.csv.gz'))

   # Get a list of all participant IDs
   pids = np.unique([os.path.basename(x).split('_')[0] for x in fitbit_aligned_files])

   data_dict = {}
   for pid in pids:
      data_dict[pid] = {}
      for fitbit_file in fitbit_aligned_files:
         if os.path.basename(fitbit_file).startswith(pid):
            fitbit_df = pd.read_csv(fitbit_file)
            data_dict[pid]['fitbit'] = fitbit_df

   return data_dict

# Returns the scalars a,b that optimize ||y - a*x - b||_2 for 1D arrays (vectors) x and y
def OptimizeLinearProcrustes(x, y):
   assert len(x) == len(y)
   n = float(len(x))
   no_nan_x = np.nan_to_num(x)
   no_nan_y = np.nan_to_num(y)
   nan_dot_xy = np.dot(no_nan_x, no_nan_y)
   nan_dot_xx = np.dot(no_nan_x, no_nan_x)
   a = (n*nan_dot_xy - np.nansum(x)*np.nansum(y))/(n*nan_dot_xx - np.nansum(x)**2)
   b = (np.nansum(y) - a*np.nansum(x))/n
   return a, b

# Input: signal (1D numpy array), dictionary (list of 1D numpy arrays)
# Output: approximate signal
def ReconstructSignalWithDictionary(signal, dictionary):
   skip_nan_percent = 0.1

   num_atoms = 0
   signal_fit = np.nan*np.zeros(len(signal))
   masked_signal = signal.copy()
   while True:
      # Reconstruct the signal using the motif dictionary
      best_dict_idx = -1
      best_dict_coefs = None
      best_dict_frame = np.nan
      best_error = np.inf
      for dict_idx in range(len(dictionary)):
         dict_vec = dictionary[dict_idx]

         for i in range(len(signal) - len(dict_vec)):
            left_idx = i
            right_idx = left_idx+len(dict_vec)-1

            # Skip mostly NaN regions
            if np.sum(np.isnan(masked_signal[left_idx:right_idx+1])) > skip_nan_percent*(right_idx-left_idx+1):
               continue

            # Find the best fit
            a, b = OptimizeLinearProcrustes(dict_vec, masked_signal[left_idx:right_idx+1])
            if np.isnan(a) or np.isnan(b) or a <= 0: # Do not allow 'flipped' dictionary fits
               continue
            #if a < 0.5 or a > 2.0: # TODO - create better bounds!
            #   continue
            best_fit = a*dict_vec + b
            residual = masked_signal[left_idx:right_idx+1] - best_fit
            np.nan_to_num(residual, copy=False) # Replace NaN with zero
            error = np.dot(residual, residual)
            if error < best_error:
               best_error = error
               best_dict_idx = dict_idx
               best_dict_coefs = (a,b)
               best_dict_frame = i

      if best_dict_idx < 0:
         print("No best next dictionary element found")
         break
      else:
         num_atoms += 1
         dict_len = len(dictionary[best_dict_idx])
         dict_signal = best_dict_coefs[0]*dictionary[best_dict_idx] + best_dict_coefs[1]
         signal_fit[best_dict_frame:best_dict_frame+dict_len] = dict_signal
         masked_signal[best_dict_frame:best_dict_frame+dict_len] = np.inf
      
      #plt.plot(range(len(signal)), signal, 'b--')
      #plt.plot(range(len(signal_fit)), signal_fit, 'r-', linewidth=2)
      #plt.title('Reconstructed signal with %d atom(s)'%(num_atoms))
      #plt.show()

   return signal_fit, np.isinf(masked_signal)

#@profile
def RunMP(aligned_data_root_path, output_path):
   min_valid_frame_ratio = 0.5
   min_routine_chain_length = 2
   max_num_motifs = 999999
   window_size = 1300
   streams = ['HeartRatePPG', 'StepCount']

   data_dict = LoadAlignedTILESData(aligned_data_root_path)
   pids = list(data_dict.keys())[2:]

   # Compute motifs from the individual MP using a greedy method
   for pid in pids:
      signal_df = data_dict[pid]['fitbit']
      #signal_df = signal_df.iloc[0:5000,:] # HACK for speed
      exclusion_mask = np.array(signal_df.shape[0]*[False])

      for stream in streams:
         signal = signal_df[stream]

         # Use Matrix Profile methods to learn a motif dictionary
         motifs = []
         while len(motifs) < max_num_motifs:
            print("Number of motifs found: %d"%(len(motifs)))
            masked_signal = signal.copy()
            masked_signal[exclusion_mask] = np.nan
            fitbit_mp = stumpy.stump(masked_signal, m=window_size)

            left_mp_idx = fitbit_mp[:,2]
            right_mp_idx = fitbit_mp[:,3]
            all_chain_set, unanchored_chain = stumpy.allc(left_mp_idx, right_mp_idx)

            is_valid_chain = True
            for i in range(unanchored_chain.shape[0]):
               num_nan_frames = np.sum(np.isnan(masked_signal[unanchored_chain[i]:unanchored_chain[i]+window_size]))
               if num_nan_frames >= min_valid_frame_ratio*window_size:
                  is_valid_chain = False
                  break

            if do_show_plot:
               nonroutine_signal = signal.copy()
               nonroutine_signal[exclusion_mask] = np.nan
               fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})
               ax[0].plot(signal_df.index, nonroutine_signal, 'b-')
               for i in range(unanchored_chain.shape[0]):
                  y = signal.iloc[unanchored_chain[i]:unanchored_chain[i]+window_size]
                  x = y.index.values
                  ax[0].plot(x, y, linewidth=3, linestyle='--')
               ax[1].plot(signal_df.index[0:len(fitbit_mp)], fitbit_mp[:,0])
               ax[1].set_ylabel('Matrix Profile')
               plt.show()

            if is_valid_chain and (unanchored_chain.shape[0] >= min_routine_chain_length):
               # TODO - Which part of the chain should be the motif?
               motif = signal.iloc[unanchored_chain[0]:unanchored_chain[0]+window_size].values
               motifs.append(motif)
               for i in range(unanchored_chain.shape[0]):
                  exclusion_mask[unanchored_chain[i]:unanchored_chain[i]+window_size] = True
            else:
               break

         routine_signal, routine_mask = ReconstructSignalWithDictionary(signal, motifs)

         fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})
         plt.suptitle('Fitbit %s with routines removed'%(stream))
         ax[0].plot(signal_df.index, signal, 'b-')
         ax[1].plot(signal_df.index, signal, 'b--')
         ax[1].plot(signal_df.index, routine_signal, 'g-', linewidth=3)
         plt.show()

   return

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--aligned_data_path', required=True, help='Path to the root folder of the aligned data')
   parser.add_argument('--output_path', required=True, help='Output CSV file path')
   try:
      args = parser.parse_args()
   except:
      parser.print_help()
      sys.exit(0)
   RunMP(args.aligned_data_path, args.output_path)
