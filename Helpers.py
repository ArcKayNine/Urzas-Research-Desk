from collections import Counter
from scipy import sparse
from pathlib import Path

import pandas as pd
import numpy as np

import json


# Load and process data
def load_data(data_path='processed_data', lookback_days=365):
    """
    Load preprocessed MTG tournament data for dashboard visualization.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing processed data files
    lookback_days : int
        Number of days of data to load (to avoid loading entire history)
        
    Returns:
    --------
    tuple
        (DataFrame with deck data, sparse matrix of card counts, fitted CountVectorizer vocabulary)
    """
    # Load the preprocessed data
    with open(Path(data_path) / 'deck_data.json', 'r') as f:
        data = json.load(f)
        
    # Convert to DataFrame
    df = pd.DataFrame(data['decks'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # Load cluster labels
    df['Cluster'] = data['clusters']

    # Filter to recent data
    cutoff_date = (pd.to_datetime('today') - pd.Timedelta(days=lookback_days)).date()

    # Load card vectors
    X = sparse.load_npz(Path(data_path) / 'card_vectors.npz')[df['Date'] >= cutoff_date]

    df = df[df['Date'] >= cutoff_date].reset_index()
    
    # Load and reconstruct vectorizer
    with open(Path(data_path) / 'vectorizer.json', 'r') as f:
        vectorizer_data = json.load(f)
    
    # vectorizer = CountVectorizer()
    # vectorizer.vocabulary_ = vectorizer_data['vocabulary']
    # vectorizer.fixed_vocabulary_ = True
    
    return df, X, vectorizer_data['vocabulary']

def sparse_column_value_counts(sparse_matrix, normalize=True):
    """
    Calculate value counts for each column in a sparse matrix without densification.
    
    Parameters:
    -----------
    sparse_matrix : scipy.sparse.spmatrix
        Input sparse matrix (will be converted to CSC format internally)
    normalize : bool, default=True
        If True, returns the relative frequency of values. If False, returns counts.
    
    Returns:
    --------
    list of dicts
        Each dict contains value:count pairs for a column.
        If normalize=True, counts are replaced with frequencies.
    """
    # Convert to CSC for efficient column access
    if not sparse.isspmatrix_csc(sparse_matrix):
        csc_matrix = sparse_matrix.tocsc()
    else:
        csc_matrix = sparse_matrix
    
    n_rows, n_cols = csc_matrix.shape
    result = []
    
    for col_idx in range(n_cols):
        # Get column data and row indices
        start = csc_matrix.indptr[col_idx]
        end = csc_matrix.indptr[col_idx + 1]
        data = csc_matrix.data[start:end]
        
        # Count explicitly stored values
        counter = Counter(data)
        
        # Add count for zeros (elements not explicitly stored)
        explicit_entries = end - start
        zeros_count = n_rows - explicit_entries
        if zeros_count > 0:
            counter[0] = zeros_count
        
        # Normalize if requested
        if normalize:
            total = n_rows
            counter = {k: v / total for k, v in counter.items()}
        
        result.append(counter)
    
    return result

def vertical_bar_html(value):
    """
    Format a tabulator with a vertical bar to produce histograms across neighbouring columns.
    Input should already be normalized to between 0,1.
    """
    if pd.isna(value):
        return ""
    
    percent = max(0, min(100, value * 100))
    
    return f"""
        <div style="margin: 0 auto; position: relative; width: 30px; height: 20px; background-color: #f0f0f0; border-radius: 3px;">
            <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: {percent}%; background-color: #6495ED; border-radius: 0 0 3px 3px;"></div>
            <div style="position: absolute; width: 100%; text-align: center; top: 50%; transform: translateY(-50%); font-size: 10px;">{percent:.0f}%</div>
        </div>
    """

def fuzzy_join(df1, df2):
    """
    Join two dataframes on 'Player' column, handling duplicate names by matching based on closest rank.
    This handles for when there are duplicate player names in an event.
    
    Parameters:
    df1, df2: Pandas DataFrames with 'Player' and 'Rank' columns
    
    Returns:
    Pandas DataFrame with joined results
    """
    # Step 1: Do a standard join on names first
    # This will work for all unique names
    standard_join = pd.merge(df1, df2, on='Player', how='inner', suffixes=('','_standings'))
    
    # Step 2: Find duplicate names from both dataframes
    duplicate_names_df1 = df1['Player'].value_counts()[df1['Player'].value_counts() > 1].index.tolist()
    duplicate_names_df2 = df2['Player'].value_counts()[df2['Player'].value_counts() > 1].index.tolist()
    duplicate_names = list(set(duplicate_names_df1 + duplicate_names_df2))
    
    # Step 3: Remove duplicate named rows from the standard join
    clean_join = standard_join[~standard_join['Player'].isin(duplicate_names)]
    
    # Step 4: Handle duplicates separately
    fuzzy_results = []
    for dup_name in duplicate_names:
        # Get all rows with this name from both dataframes
        dup_df1 = df1[df1['Player'] == dup_name].copy()
        dup_df2 = df2[df2['Player'] == dup_name].copy()
        
        # If we have duplicates in both dataframes, we need to do fuzzy matching
        if len(dup_df1) > 0 and len(dup_df2) > 0:
            # Create a distance matrix between all rank combinations
            distances = np.zeros((len(dup_df1), len(dup_df2)))
            
            for i, row1 in enumerate(dup_df1.itertuples()):
                for j, row2 in enumerate(dup_df2.itertuples()):
                    distances[i, j] = abs(row1.Rank - row2.Rank)
            
            # Match rows greedily by minimum rank distance
            matched_pairs = []
            while len(matched_pairs) < min(len(dup_df1), len(dup_df2)):
                # Find the minimum distance
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                matched_pairs.append((min_idx[0], min_idx[1]))
                
                # Mark this pair as matched by setting distance to infinity
                distances[min_idx[0], :] = np.inf
                distances[:, min_idx[1]] = np.inf
            
            # Create joined rows based on matched pairs
            for df1_idx, df2_idx in matched_pairs:
                row_df1 = dup_df1.iloc[df1_idx]
                row_df2 = dup_df2.iloc[df2_idx]
                
                joined_row = pd.DataFrame({
                    'name': [row_df1['Player']],
                    'rank_df1': [row_df1['Rank']],
                    'rank_df2': [row_df2['Rank']]
                })
                
                fuzzy_results.append(joined_row)
    
    # Step 5: Combine standard join with fuzzy results
    if fuzzy_results:
        fuzzy_join = pd.concat(fuzzy_results, ignore_index=True)
        final_result = pd.concat([clean_join, fuzzy_join], ignore_index=True)
    else:
        final_result = clean_join
    
    return final_result