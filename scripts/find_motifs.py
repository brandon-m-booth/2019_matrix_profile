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
from scipy.stats import pearsonr, spearmanr, kendalltau

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
# Output: approximate signal (1D array_, reconstruction data (list of tuples containing  motif data used to generate the approximate signal: (dict idx, frame offset, atom scale factor, atom shift amount) )
def ReconstructSignalWithDictionary(signal, dictionary):
   skip_nan_percent = 0.1

   if len(dictionary) == 0:
      return (np.nan*np.zeros(len(signal)), [])

   # Compute and cache the optimzal fit coefficients for each atom and window
   print("Fitting coefficients for each atom")
   min_atom_len = min([len(x) for x in dictionary])
   fit_data = np.inf*np.ones((len(dictionary), len(signal)-min_atom_len, 4)) # Format: dict_idx, window_idx, (a,b,num nan,error)
   for dict_idx in range(len(dictionary)):
      dict_vec = dictionary[dict_idx]
      for i in range(len(signal) - len(dict_vec)):
         left_idx = i
         right_idx = left_idx+len(dict_vec)-1
         a, b = OptimizeLinearProcrustes(dict_vec, signal[left_idx:right_idx+1])
         num_nan = np.sum(np.isnan(signal[left_idx:right_idx+1]))
         residual = signal[left_idx:right_idx+1] - (a*dict_vec+b)
         np.nan_to_num(residual, copy=False)
         error = np.dot(residual, residual)/(len(dict_vec)) # MSE
         fit_data[dict_idx, i, :] = (a, b, num_nan, error)
   
   valid_mask = np.zeros((fit_data.shape[0], fit_data.shape[1])).astype(bool)
   for dict_idx in range(len(dictionary)):
      dict_len = len(dictionary[dict_idx])
      nan_mask = fit_data[dict_idx,:,2] < skip_nan_percent*dict_len
      coef_mask = fit_data[dict_idx,:,0] >= 0
      valid_mask[dict_idx,:] = np.logical_and(nan_mask, coef_mask)

   masked_error_data = fit_data[:,:,3].copy()
   masked_error_data[~valid_mask] = np.inf
   signal_fit = np.nan*np.zeros(len(signal))
   reconstruction_data = []
   num_atoms = 0
   while True:
      min_error_idx = np.argmin(masked_error_data)
      min_error_idx = np.unravel_index(min_error_idx, masked_error_data.shape)
      if not np.isinf(masked_error_data[min_error_idx]):
         num_atoms += 1
         dict_len = len(dictionary[min_error_idx[0]])
         a = fit_data[min_error_idx[0], min_error_idx[1], 0]
         b = fit_data[min_error_idx[0], min_error_idx[1], 1]
         dict_signal = a*dictionary[min_error_idx[0]]+b
         signal_fit[min_error_idx[1]:min_error_idx[1]+dict_len] = dict_signal

         reconstruction_data.append((min_error_idx[0], min_error_idx[1], a, b))

         #print("--------------------")
         #print("Min idx: (%d,%d)"%(min_error_idx))
         #print("A: %f, B: %f"%(a, b))
         #plt.plot(range(len(signal)), signal, 'b--')
         #plt.plot(range(len(signal_fit)), signal_fit, 'r-', linewidth=2)
         #plt.title('Reconstructed signal with %d atom(s)'%(num_atoms))
         #plt.show()

         # Update the masked error data to prevent atom overlap
         for i in range(len(dictionary)):
            dict_len = len(dictionary[i])
            left_mask_idx = max(0, min_error_idx[1]-dict_len+1)
            right_mask_idx = min(fit_data.shape[1], min_error_idx[1]+dict_len-1)
            masked_error_data[i, left_mask_idx:right_mask_idx+1] = np.inf
      else:
         break
      
   return signal_fit, reconstruction_data

def ExtractFeatures(routine_signal, signal, motifs, motifs_data, reconstruction_data):
   # Routineness (binary, scale factor and shift, duplicity, reconstruction correlation (pearson, spearman, kendall's tau)
   feature_cols = ['routineness_binary', 'routineness_scale', 'routineness_shift', 'routineness_duplicity', 'routineness_pearson', 'routineness_spearman', 'routineness_kendall_tau']
   features_df = pd.DataFrame(data=np.zeros((signal.shape[0], len(feature_cols))), index=signal.index, columns=feature_cols)

   for recon_idx in range(len(reconstruction_data)):
      (motif_idx, frame_offset, atom_scale, atom_shift) = reconstruction_data[recon_idx]
      motif = motifs[motif_idx]
      chain = motifs_data[motif_idx]
      left_index = frame_offset+signal.index[0]
      right_index = left_index+len(motif)-1
      routine_signal_subseq = routine_signal[left_index:right_index+1]
      signal_subseq = signal[left_index:right_index+1]
      routine_invalid_mask = np.logical_or(np.isnan(routine_signal_subseq), np.isinf(routine_signal_subseq))
      signal_invalid_mask = np.logical_or(np.isnan(signal_subseq), np.isinf(signal_subseq))
      valid_mask = ~np.logical_or(routine_invalid_mask, signal_invalid_mask)
      routine_signal_subseq_valid = routine_signal_subseq[valid_mask]
      signal_subseq_valid = signal_subseq[valid_mask]

      routineness_binary = 1
      routineness_scale = atom_scale
      routineness_shift = atom_shift
      routineness_duplicity = chain.shape[0]
      routineness_pearson = pearsonr(signal_subseq_valid, routine_signal_subseq_valid)[0]
      routineness_spearman = spearmanr(signal_subseq_valid, routine_signal_subseq_valid)[0]
      routineness_kendalltau = kendalltau(signal_subseq_valid, routine_signal_subseq_valid)[0]

      features_df.loc[left_index:right_index+1,:] = [routineness_binary, routineness_scale, routineness_shift, routineness_duplicity, routineness_pearson, routineness_spearman, routineness_kendalltau]

   return features_df

#@profile
def RunMP(aligned_data_root_path, output_path):
   min_valid_frame_ratio = 0.5
   min_routine_chain_length = 2
   max_num_motifs = 999999
   window_size = 1300
   streams = ['HeartRatePPG', 'StepCount']

   if not os.path.isdir(output_path):
      os.makedirs(output_path)

   data_dict = LoadAlignedTILESData(aligned_data_root_path)
   pids = list(data_dict.keys())

   # Compute motifs from the individual MP using a greedy method
   for pid in pids:
      signal_df = data_dict[pid]['fitbit']
      #signal_df = signal_df.iloc[0:5000,:] # HACK for speed
      exclusion_mask = np.array(signal_df.shape[0]*[False])

      for stream in streams:
         signal = signal_df[stream]

         # Use Matrix Profile methods to learn a motif dictionary
         motifs = []
         motifs_data = []
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
               motifs_data.append(unanchored_chain)
               for i in range(unanchored_chain.shape[0]):
                  exclusion_mask[unanchored_chain[i]:unanchored_chain[i]+window_size] = True
            else:
               break

         routine_signal, reconstruction_data = ReconstructSignalWithDictionary(signal, motifs)

         features_df = ExtractFeatures(routine_signal, signal, motifs, motifs_data, reconstruction_data)
         features_df.to_csv(os.path.join(output_path, '%s_routine_features.csv'%(pid)), index=False, header=True)

         if do_show_plot:
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
