# process_data.py
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import json
from pathlib import Path
from datetime import datetime, timedelta
import umap
import hdbscan
from vectorizers.transformers import InformationWeightTransformer
from tqdm import tqdm

def get_tournament_files(base_path='../MTGODecklistCache/Tournaments', lookback_days=365, fmt='modern'):
    """
    Find all modern tournament files from the last lookback_days.
    
    Parameters:
    -----------
    base_path : str
        Path to tournament data directory
    lookback_days : int
        Number of days to look back
    fmt : str
        Tournament format
        
    Returns:
    --------
    list
        List of Path objects for matching tournament files
    """
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    # Get all possible year/month/day combinations from cutoff to now
    date_range = []
    current_date = cutoff_date
    while current_date <= datetime.now():
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Create patterns for each date
    patterns = [
        f"*/{date.year}/{date.month:02d}/{date.day:02d}/*{fmt}*.json"
        for date in date_range
    ]
    
    # Find all matching files
    matching_files = []
    base_path = Path(base_path)
    for pattern in patterns:
        matching_files.extend(base_path.glob(pattern))
    
    return matching_files

def process_mtg_data(lookback_days=365, fmt='modern'):
    """Process MTG tournament data and save results for dashboard consumption."""
    
    # Initialize empty DataFrame
    df = pd.DataFrame()
    
    # Process tournament files
    print('Tournament Files')
    tournament_path = Path('../MTGODecklistCache/Tournaments/')
    for path in tqdm(get_tournament_files(tournament_path, lookback_days, fmt)):
        with open(path) as f:
            data = json.load(f)
        
        deck_df = pd.DataFrame(data['Decks'])
        deck_df['Deck'] = data['Decks']
        deck_df['Tournament'] = path.name
        
        # Process standings
        standings_df = pd.DataFrame(data['Standings'])
        if standings_df.shape[0]:
            deck_df = deck_df.join(standings_df.set_index('Player'), on='Player')
            deck_df['League'] = False
        else:
            deck_df['Wins'] = 5
            deck_df['Losses'] = 0
            deck_df['League'] = True

        
        # Set date from path if missing
        deck_df['Date'] = f'{path.parent.parent.parent.name}-{path.parent.parent.name}-{path.parent.name}'
        
        df = pd.concat([df, deck_df], ignore_index=True)
    
    # Convert dates and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    print(f'deck data loaded, shape={df.shape}')

    # Load card data
    with open('../AtomicCards.json', 'r') as f:
        j = json.load(f)['data']
    card_list = j.keys()

    print(f'card data loaded, shape={len(card_list)}')
    
    # Vectorize decks
    def merge_analyzer(deck):
        """Convert deck dictionary into list of card strings."""
        output = []
        for card in deck['Mainboard']:
            # if card['CardName'] in card_list:
            #     if 'Land' not in j[card['CardName']][0]['type']:
            #         output += [card['CardName']] * card['Count']
            # else:
            output += [card['CardName']] * card['Count']
        for card in deck['Sideboard']:
            output += [card['CardName']+'_SB'] * card['Count']
        return output

    vectorizer = CountVectorizer(analyzer=merge_analyzer)
    X = vectorizer.fit_transform(df['Deck'])
    
    # Apply Information Weight Transform
    iwt = InformationWeightTransformer()
    X_iwt = iwt.fit_transform(X)

    print('Vectorized')
    
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(
        n_components=3,
        metric='cosine',
        # n_neighbors=15,
        # min_dist=0.1,
        random_state=42
    )
    
    X_umap = reducer.fit_transform(X_iwt)

    print('UMAP complete')
    
    # Perform clustering on UMAP embedding
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=5,
        cluster_selection_epsilon=0.5,
        metric='euclidean'  # Use euclidean for reduced dimensions
    )
    
    cluster_labels = clusterer.fit_predict(X_umap)

    print(f'HDBSCAN complete, {clusterer.labels_.max()} clusters')
    
    # Calculate cluster representatives
    cluster_representatives = {}
    for label in tqdm(range(clusterer.labels_.max()+1)):
        # Get decks in this cluster
        cluster_mask = cluster_labels == label
        cluster_vectors = X[cluster_mask]
        
        # Calculate mean card counts
        mean_counts = cluster_vectors.mean(axis=0).A1
        std_counts = cluster_vectors.toarray().std(axis=0)
        
        # Get top cards by mean/std ratio
        card_stats = list(zip(
            mean_counts,
            std_counts,
            vectorizer.get_feature_names_out()
        ))
        
        # Sort by mean/std ratio, handling divide by zero
        top_cards = sorted([
            (m, s, n) for m, s, n in card_stats
            if m > 0.1  # Only include cards that appear in at least 10% of decks
        ], key=lambda x: x[0]/(x[1] if x[1] > 0 else 0.1), reverse=True)[:10]
        
        cluster_representatives[label] = {
            'size': int(cluster_mask.sum()),
            'win_rate': float(df.loc[cluster_mask, 'Wins'].sum() / 
                            (df.loc[cluster_mask, 'Wins'].sum() + df.loc[cluster_mask, 'Losses'].sum())),
            'top_cards': [{'name': n, 'mean': float(m), 'std': float(s)} for m, s, n in top_cards]
        }

    print('Clusters analysed')
    
    # Create output directory
    Path('processed_data').mkdir(exist_ok=True)

    # Generate and save metadata
    metadata = {
        'last_updated': datetime.utcnow().isoformat(),
        'num_decks': df.shape[0],
        'date_range': [df['Date'].min().isoformat(), df['Date'].max().isoformat()],
        'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    }
    
    with open('processed_data/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    df['Date'] = df['Date'].astype(str)
    # Save processed data
    output_data = {
        'decks': df[['Player', 'Wins', 'Losses', 'Date', 'Tournament', 'League']].to_dict('records'),
        'clusters': cluster_labels.tolist(),
        'cluster_info': cluster_representatives,
        'feature_names': vectorizer.get_feature_names_out().tolist()
    }
    
    with open('processed_data/deck_data.json', 'w') as f:
        json.dump(output_data, f)
    
    # Save matrices
    scipy.sparse.save_npz('processed_data/card_vectors.npz', X)
    np.save('processed_data/umap_embedding.npy', X_umap)
    
    # Save transformers data
    vectorizer_data = {
        'vocabulary': vectorizer.vocabulary_
    }
    with open('processed_data/vectorizer.json', 'w') as f:
        json.dump(vectorizer_data, f)
        
    # iwt_data = {
    #     'idf': iwt.idf_.tolist()
    # }
    # with open('processed_data/iwt.json', 'w') as f:
    #     json.dump(iwt_data, f)

    print('Data saved, done')

if __name__ == '__main__':
    process_mtg_data()#lookback_days=30)