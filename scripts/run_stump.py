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
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

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

def RunMP(aligned_data_root_path, output_path):
   do_compute_individual_k_motifs = True
   do_compute_anchored_chains = False
   do_compute_semantic_segmentation = False
   do_compute_multimodal_mp = False
   window_size = 1500
   #window_size = 1500
   data_dict = LoadAlignedTILESData(aligned_data_root_path)

   #plt.ion()

   pids = list(data_dict.keys())[0:1]
   streams = ['HeartRatePPG', 'StepCount']

   # Compute motifs from the individual MP using a greedy method
   if do_compute_individual_k_motifs:
      num_motifs = 1
      for pid in pids:
         fitbit_df = data_dict[pid]['fitbit']
         fitbit_df = fitbit_df.iloc[0:4000,:] # HACK

         for stream in streams:
            # Keep a NaN'd version for MP and interpolated one for OMP
            #nan_replace_value = -1000000
            #fitbit_df[stream][np.isnan(fitbit_df[stream])] = nan_replace_value
            #fitbit_df_smooth = fitbit_df[stream].interpolate(method='linear', axis=0, inplace=False)
            fitbit_df_smooth = fitbit_df[stream].copy()

            if np.isnan(fitbit_df_smooth[0]): # Fill NaNs at the beginning and end
               idx = 0
               while np.isnan(fitbit_df_smooth[idx]):
                  idx += 1
               fitbit_df_smooth[0:idx] = fitbit_df_smooth[idx]
            if np.isnan(fitbit_df_smooth[fitbit_df_smooth.shape[0] - 1]):
               idx = fitbit_df_smooth.shape[0] - 1
               while np.isnan(fitbit_df_smooth[idx]):
                  idx -= 1
               fitbit_df_smooth[idx:] = fitbit_df_smooth[idx]

            # Use Matrix Profile methods to learn a motif dictionary
            motifs = []
            while len(motifs) < num_motifs:
               fitbit_mp = stumpy.stump(fitbit_df[stream], m=window_size)
               fitbit_mp_argsort = np.array(fitbit_mp[:,0]).argsort()
               for motif_idx in range(len(fitbit_mp_argsort)):
                  stream_motif_idx = fitbit_mp_argsort[motif_idx]
                  num_nan = np.sum(np.isnan(fitbit_df[stream].values[stream_motif_idx:stream_motif_idx+window_size]))

                  # Avoid finding bad motifs
                  if num_nan >= 5.0*window_size/6.0:
                     continue
                  if stream == 'HeartRatePPG':
                     pass
                  break
               motif_left_idx = fitbit_mp_argsort[motif_idx]
               motif = fitbit_df_smooth[motif_left_idx:motif_left_idx+window_size]
               motif[motif == 0] = 1e-12 # OMP requires non-zeros in the support
               motifs.append(motif)
               plt.plot(range(motif_left_idx,motif_left_idx+window_size), motifs[-1], 'g-', linewidth=5)

            # Build a redundant dictionary from the motifs
            num_repetitions = len(fitbit_df_smooth)-window_size
            dictionary_mat = np.zeros((len(motifs)*num_repetitions,len(fitbit_df_smooth)))
            for motif_idx in range(len(motifs)):
               for repeat_idx in range(num_repetitions):
                  dictionary_mat[motif_idx*num_repetitions + repeat_idx, repeat_idx:repeat_idx+window_size] = motifs[motif_idx].values

            # Reconstruct the signal using the motif dictionary
            # TODO : Write my own OMP with exclusion of each atom's support. Gram mat?
            # TODO : Use L1 optimization (Lasso)?
            #omp = OrthogonalMatchingPursuit(n_nonzero_coefs=2, fit_intercept=False)
            #omp = OrthogonalMatchingPursuitCV(fit_intercept=False)
            #omp.fit(dictionary_mat.T, fitbit_df_smooth)
            #intercept = omp.intercept_
            #coef = omp.coef_
            #idx_r = coef.nonzero()
            #num_nonzero = omp.n_nonzero_coefs_

            max_nonzero = 4
            coef = np.zeros((dictionary_mat.T.shape[1],1))
            intercept = np.zeros((dictionary_mat.T.shape[0],1))
            exclusion_signal = fitbit_df[stream].copy()
            for num_nonzero in range(1,max_nonzero+1):
               # Reconstruct the signal using the motif dictionary
               best_dict_idx = -1
               best_error = np.inf
               best_dict_support = None
               for dict_idx in range(dictionary_mat.shape[0]):
                  dict_vec = dictionary_mat[dict_idx,:]

                  # Find the support
                  left_support_idx = 0
                  right_support_idx = len(dict_vec)-1
                  while dict_vec[left_support_idx] == 0 and left_support_idx < len(dict_vec):
                     left_support_idx += 1
                  while dict_vec[right_support_idx] == 0 and right_support_idx >= 0:
                     right_support_idx -= 1

                  # Find the best match
                  residual = exclusion_signal[left_support_idx:right_support_idx+1] - dict_vec[left_support_idx:right_support_idx+1]
                  np.nan_to_num(residual, copy=False) # Replace NaN with zero
                  error = np.dot(residual, residual)
                  if error < best_error:
                     best_error = error
                     coef_val = 1 # TODO - constrain between 0.5 and 2?
                     best_dict_idx = dict_idx
                     best_dict_support = (left_support_idx, right_support_idx)

               if best_dict_idx < 0:
                  print("No best next dictionary element found")
                  break

               # Update coef
               coef_nonzero = (coef != 0).reshape(-1,)
               if len(coef_nonzero) > 0:
                  dictionary_mat_reduced = dictionary_mat[coef_nonzero, :]
                  coef_reduced = coef[coef_nonzero]

                  #prev_fit_signal = np.matmul(dictionary_mat.T, coef)
                  prev_fit_signal = np.matmul(dictionary_mat_reduced.T, coef_reduced)
                  prev_residual = fitbit_df_smooth - prev_fit_signal.reshape(-1,)
               else:
                  prev_residual = fitbit_df_smooth- np.zeros(len(fitbit_df_smooth))
               np.nan_to_num(prev_residual, copy=False) # Replace NaN with zero
               prev_error = np.dot(prev_residual, prev_residual)

               coef[best_dict_idx] = coef_val
               #fit_signal = np.matmul(dictionary_mat.T, coef)
               fit_signal = np.matmul(dictionary_mat_reduced.T, coef_reduced)
               fit_residual = fitbit_df_smooth - prev_fit_signal.reshape(-1,)
               np.nan_to_num(fit_residual, copy=False) # Replace NaN with zero
               fit_error = np.dot(fit_residual, fit_residual)

               if best_dict_support is not None:
                  exclusion_signal[best_dict_support[0]:best_dict_support[1]+1] = np.inf

               if prev_error < fit_error:
                  print("Avoiding overfitting...")
                  coef[best_dict_idx,0] = 0
                  break

            coef_nonzero = (coef != 0).reshape(-1,)
            dictionary_mat_reduced = dictionary_mat[coef_nonzero, :]
            coef_reduced = coef[coef_nonzero]
            fit_signal = np.matmul(dictionary_mat_reduced.T, coef_reduced) + intercept
            plt.plot(range(fitbit_df[stream].shape[0]), fitbit_df[stream], 'b-')
            #plt.plot(range(fitbit_df_smooth.shape[0]), fitbit_df_smooth, 'k-')
            plt.plot(range(fitbit_df[stream].shape[0]), fit_signal, 'r--')
            plt.title('OMP (%d coefs) + MP Motifs (%d motifs)'%(num_nonzero, num_motifs))
            plt.xlabel('Time')
            plt.ylabel(stream)
            plt.show()
            pdb.set_trace()

   # Compute individual matrix profiles (stump)
   if do_compute_anchored_chains or do_compute_semantic_segmentation:
      for pid in pids:
         fitbit_df = data_dict[pid]['fitbit']
         for stream in streams:
            fitbit_mp = stumpy.stump(fitbit_df[stream], m=window_size)

            if do_compute_anchored_chains:
               left_mp_idx = fitbit_mp[:,2]
               right_mp_idx = fitbit_mp[:,3]
               #atsc_idx = 10
               #anchored_chain = stumpy.atsc(left_mp_idx, right_mp_idx, atsc_idx)
               all_chain_set, unanchored_chain = stumpy.allc(left_mp_idx, right_mp_idx)

            if do_compute_semantic_segmentation:
               subseq_len = window_size
               correct_arc_curve, regime_locations = stumpy.fluss(fitbit_mp[:,1], L=subseq_len, n_regimes=2, excl_factor=5)

            # Find the first motif with nearly no NaN values in the stream signal
            fitbit_mp_argsort = np.array(fitbit_mp[:,0]).argsort()
            for motif_idx in range(len(fitbit_mp_argsort)):
               stream_motif_idx = fitbit_mp_argsort[motif_idx]
               num_nan = np.sum(np.isnan(fitbit_df[stream].values[stream_motif_idx:stream_motif_idx+window_size]))

               # Avoid finding bad motifs
               if num_nan >= 5.0*window_size/6.0:
                  continue
               if stream == 'HeartRatePPG':
                  pass
                  # Check for flat heart rate
                  #nan_like_value = 70
                  #num_valid = np.count_nonzero((fitbit_df[stream] - nan_like_value)[stream_motif_idx:stream_motif_idx+window_size])
                  #if num_valid < window_size - 2:
                  #   continue
                  
                  # Check for linear heart rate over time
                  #residual_threshold = window_size*(4.0**2)
                  #p, res, rank, sing_vals, rcond = np.polyfit(range(window_size), fitbit_df[stream][stream_motif_idx:stream_motif_idx+window_size], deg=1, full=True)
                  #if res < residual_threshold:
                  #   continue
               break

            num_subplots = 3 if do_compute_semantic_segmentation else 2
            fig,axs = plt.subplots(num_subplots, sharex=True, gridspec_kw={'hspace':0})
            plt.suptitle('Matrix Profile, %s, PID: %s'%(stream, pid), fontsize='30')
            axs[0].plot(fitbit_df[stream].values)
            rect = plt.Rectangle((fitbit_mp_argsort[motif_idx],0), window_size, 2000, facecolor='lightgrey')
            axs[0].add_patch(rect)
            rect = plt.Rectangle((fitbit_mp_argsort[motif_idx+1],0), window_size, 2000, facecolor='lightgrey')
            axs[0].add_patch(rect)
            axs[0].set_ylabel(stream, fontsize='20')
            axs[1].plot(fitbit_mp[:,0])
            axs[1].axvline(x=fitbit_mp_argsort[motif_idx], linestyle="dashed")
            axs[1].axvline(x=fitbit_mp_argsort[motif_idx+1], linestyle="dashed")
            axs[1].set_ylabel('Matrix Profile', fontsize='20')

            if do_compute_anchored_chains:
               for i in range(unanchored_chain.shape[0]):
                  y = fitbit_df[stream].iloc[unanchored_chain[i]:unanchored_chain[i]+window_size]
                  x = y.index.values
                  axs[0].plot(x, y, linewidth=3)

            if do_compute_semantic_segmentation:
               axs[2].plot(range(correct_arc_curve.shape[0]), correct_arc_curve, color='C1')
               axs[0].axvline(x=regime_locations[0], linestyle="dashed")
               axs[2].axvline(x=regime_locations[0], linestyle="dashed")

            plt.show()

   # Compute multi-dimensional matrix profiles (mstump)
   if do_compute_multimodal_mp:
      for pid in pids:
         fitbit_df = data_dict[pid]['fitbit']
         data = fitbit_df.loc[:,streams].values
         mp, mp_indices = stumpy.mstump(data.T, m=window_size)
         #print("Stumpy's mstump function does not handle NaN values. Skipping multi-dimensional MP")
         #break

         # TODO - This code is copied from above. Fix and finish it once mstump supports NaN
         # Find the first motif with nearly no NaN values in the stream signal
         fitbit_mp_argsort = np.array(fitbit_mp[:,0]).argsort()
         for motif_idx in range(len(fitbit_mp_argsort)):
            stream_motif_idx = fitbit_mp_argsort[motif_idx]
            num_nan = np.sum(np.isnan(fitbit_df[stream].values[stream_motif_idx:stream_motif_idx+window_size]))

            # Avoid finding bad motifs
            if num_nan >= 2:
               continue
            if stream == 'HeartRatePPG':
               # Check for flat heart rate
               nan_like_value = 70
               num_valid = np.count_nonzero((fitbit_df[stream] - nan_like_value)[stream_motif_idx:stream_motif_idx+window_size])
               if num_valid < window_size - 2:
                  continue
               
               # Check for linear heart rate over time
               residual_threshold = window_size*(4.0**2)
               p, res, rank, sing_vals, rcond = np.polyfit(range(window_size), fitbit_df[stream][stream_motif_idx:stream_motif_idx+window_size], deg=1, full=True)
               if res < residual_threshold:
                  continue
            break

         fig,axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace':0})
         plt.suptitle('Matrix Profile, %s, PID: %s'%(stream, pid), fontsize='30')
         axs[0].plot(fitbit_df[stream].values)
         rect = plt.Rectangle((fitbit_mp_argsort[motif_idx],0), window_size, 2000, facecolor='lightgrey')
         axs[0].add_patch(rect)
         rect = plt.Rectangle((fitbit_mp_argsort[motif_idx+1],0), window_size, 2000, facecolor='lightgrey')
         axs[0].add_patch(rect)
         axs[0].set_ylabel(stream, fontsize='20')
         axs[1].plot(fitbit_mp[:,0])
         axs[1].axvline(x=fitbit_mp_argsort[motif_idx], linestyle="dashed")
         axs[1].axvline(x=fitbit_mp_argsort[motif_idx+1], linestyle="dashed")
         axs[1].set_ylabel('Matrix Profile', fontsize='20')
         plt.show()

   plt.ioff()
   plt.figure()
   plt.plot()
   plt.title('Dummy plot')
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
