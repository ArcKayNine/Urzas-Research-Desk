# process_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


from tcg_research_desk import process_mtg_data
from tcg_research_desk.process_data import get_last_standard_change

# def fuzzy_join(df1, df2):
#     """
#     Join two dataframes on 'Player' column, handling duplicate names by matching based on closest rank.
#     This handles for when there are duplicate player names in an event.
    
#     Parameters:
#     df1, df2: Pandas DataFrames with 'Player' and 'Rank' columns
    
#     Returns:
#     Pandas DataFrame with joined results
#     """
#     # Step 1: Do a standard join on names first
#     # This will work for all unique names
#     standard_join = pd.merge(df1, df2, on='Player', how='inner', suffixes=('','_standings'))
    
#     # Step 2: Find duplicate names from both dataframes
#     duplicate_names_df1 = df1['Player'].value_counts()[df1['Player'].value_counts() > 1].index.tolist()
#     duplicate_names_df2 = df2['Player'].value_counts()[df2['Player'].value_counts() > 1].index.tolist()
#     duplicate_names = list(set(duplicate_names_df1 + duplicate_names_df2))
    
#     # Step 3: Remove duplicate named rows from the standard join
#     clean_join = standard_join[~standard_join['Player'].isin(duplicate_names)]
    
#     # Step 4: Handle duplicates separately
#     fuzzy_results = []
#     for dup_name in duplicate_names:
#         # Get all rows with this name from both dataframes
#         dup_df1 = df1[df1['Player'] == dup_name].copy()
#         dup_df2 = df2[df2['Player'] == dup_name].copy()
        
#         # If we have duplicates in both dataframes, we need to do fuzzy matching
#         if len(dup_df1) > 0 and len(dup_df2) > 0:
#             # Create a distance matrix between all rank combinations
#             distances = np.zeros((len(dup_df1), len(dup_df2)))
            
#             for i, row1 in enumerate(dup_df1.itertuples()):
#                 for j, row2 in enumerate(dup_df2.itertuples()):
#                     distances[i, j] = abs(row1.Rank - row2.Rank)
            
#             # Match rows greedily by minimum rank distance
#             matched_pairs = []
#             while len(matched_pairs) < min(len(dup_df1), len(dup_df2)):
#                 # Find the minimum distance
#                 min_idx = np.unravel_index(distances.argmin(), distances.shape)
#                 matched_pairs.append((min_idx[0], min_idx[1]))
                
#                 # Mark this pair as matched by setting distance to infinity
#                 distances[min_idx[0], :] = np.inf
#                 distances[:, min_idx[1]] = np.inf
            
#             # Create joined rows based on matched pairs
#             for df1_idx, df2_idx in matched_pairs:
#                 row_df1 = dup_df1.iloc[df1_idx]
#                 row_df2 = dup_df2.iloc[df2_idx]
                
#                 joined_row = pd.DataFrame({
#                     'name': [row_df1['Player']],
#                     'rank_df1': [row_df1['Rank']],
#                     'rank_df2': [row_df2['Rank']]
#                 })
                
#                 fuzzy_results.append(joined_row)
    
#     # Step 5: Combine standard join with fuzzy results
#     if fuzzy_results:
#         fuzzy_join = pd.concat(fuzzy_results, ignore_index=True)
#         final_result = pd.concat([clean_join, fuzzy_join], ignore_index=True)
#     else:
#         final_result = clean_join
    
#     return final_result

# def get_tournament_files(base_path='../MTG_decklistcache/Tournaments', lookback_days=365, fmt='modern'):
#     """
#     Find all modern tournament files from the last lookback_days.
    
#     Parameters:
#     -----------
#     base_path : str
#         Path to tournament data directory
#     lookback_days : int
#         Number of days to look back
#     fmt : str
#         Tournament format
        
#     Returns:
#     --------
#     list
#         List of Path objects for matching tournament files
#     """
#     cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
#     # Get all possible year/month/day combinations from cutoff to now
#     date_range = []
#     current_date = cutoff_date
#     while current_date <= datetime.now():
#         date_range.append(current_date)
#         current_date += timedelta(days=1)
    
#     # Create patterns for each date
#     # TODO Remove pre-modern and premodern from modern
#     #
#     patterns = [
#         f"*/{date.year}/{date.month:02d}/{date.day:02d}/*-{fmt}*.json"
#         for date in date_range
#     ]
    
#     # Find all matching files
#     matching_files = []
#     base_path = Path(base_path)
#     for pattern in patterns:
#         matching_files.extend(base_path.glob(pattern))

#     if not matching_files:
#         raise ValueError('No valid file paths were found.')
    
#     return matching_files

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("format", help="Format to process", default='Modern')
    args = parser.parse_args()

    if args.format == "Standard":
        last_date, lookback_days = get_last_standard_change()
    else:
        lookback_days = 100

    process_mtg_data(fmt=args.format, lookback_days=lookback_days)
