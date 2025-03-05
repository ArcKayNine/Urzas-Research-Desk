
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import holoviews as hv
import panel as pn

import json

import param

pn.extension('tabulator', sizing_mode="stretch_width", throttled=True)
hv.extension('bokeh')

from scipy import sparse
from scipy.stats import binomtest

# TODO
# Card image hover
# Reset selections
# Fix selection info
# Double check interaction between selection and individual card analysis
# Archetypes
# Remove mirrors
# Temporal

class MTGAnalyzer(param.Parameterized):
    selected_cards = param.List(default=[], doc="Cards required in deck")
    excluded_cards = param.List(default=[], doc="Cards excluded from deck")
    cluster_view = param.Boolean(default=True, doc="Show cluster view instead of card presence view")
    show_correlation = param.Boolean(default=False, doc="Show correlation heatmap for selected card")
    selected_card = param.String(default='', doc="Card to analyze in detail")
    date_range = param.DateRange(default=None, doc="Date range for analysis")
    valid_rows = param.Array(default=np.array([]), doc="Selected indices")
    valid_wr_rows = param.Array(default=np.array([]), doc="Selected indices with valid wr")
    
    def __init__(self, df, card_vectors, vocabulary, **params):
        super().__init__(**params)
        self.df = df
        self.X = card_vectors
        self.feature_names = vocabulary
        self._initialize_card_list()
        self.find_valid_rows()
        
    def _initialize_card_list(self):
        # Get unique cards from feature names, removing _SB suffix
        self.card_options = sorted(list(set(
            [name.replace('_SB', '') for name in self.feature_names.keys()]
        )))

    @param.depends('date_range', 'selected_cards', 'excluded_cards', watch=True)
    def find_valid_rows(self):
        """
        Find row indices where specified logical combinations of column pairs exist.
        """

        row_mask = np.ones(self.X.shape[0], dtype=bool)
        
        for pair_idx, pair in enumerate(
            [(self.feature_names.get(c), self.feature_names.get(f"{c}_SB")) for c in self.selected_cards]
            # + [(self.feature_names.get(self.selected_card), self.feature_names.get(f"{self.selected_card}_SB"))]
        ):
            # Check pair format
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each pair must be a tuple/list of length 2. Error in pair {pair_idx}: {pair}")
            
            col1, col2 = pair
            pair_rows = set()
            
            # Handle col1
            if col1 is not None:
                if not isinstance(col1, (int, np.integer)):
                    raise TypeError(f"Column index must be integer or None. Got {type(col1)} for column 1 in pair {pair_idx}")
                if col1 < 0 or col1 >= self.X.shape[1]:
                    raise ValueError(f"Column index {col1} out of bounds for matrix with {self.X.shape[1]} columns")
                pair_rows.update(self.X.getcol(col1).nonzero()[0])
                
            # Handle col2
            if col2 is not None:
                if not isinstance(col2, (int, np.integer)):
                    raise TypeError(f"Column index must be integer or None. Got {type(col2)} for column 2 in pair {pair_idx}")
                if col2 < 0 or col2 >= self.X.shape[1]:
                    raise ValueError(f"Column index {col2} out of bounds for matrix with {self.X.shape[1]} columns")
                pair_rows.update(self.X.getcol(col2).nonzero()[0])
            
            # If both columns in a pair are None, skip this pair
            if col1 is None and col2 is None:
                continue
                
            # Create mask for current pair
            current_mask = np.zeros(self.X.shape[0], dtype=bool)
            current_mask[list(pair_rows)] = True
            
            # Update overall mask (AND condition)
            row_mask &= current_mask
            
        for pair_idx, pair in enumerate(
            [(self.feature_names.get(c), self.feature_names.get(f"{c}_SB")) for c in self.excluded_cards]
        ):
            # Check pair format
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each pair must be a tuple/list of length 2. Error in pair {pair_idx}: {pair}")
            
            col1, col2 = pair
            pair_rows = set()
            
            # Handle col1
            if col1 is not None:
                if not isinstance(col1, (int, np.integer)):
                    raise TypeError(f"Column index must be integer or None. Got {type(col1)} for column 1 in pair {pair_idx}")
                if col1 < 0 or col1 >= self.X.shape[1]:
                    raise ValueError(f"Column index {col1} out of bounds for matrix with {self.X.shape[1]} columns")
                pair_rows.update(self.X.getcol(col1).nonzero()[0])
                
            # Handle col2
            if col2 is not None:
                if not isinstance(col2, (int, np.integer)):
                    raise TypeError(f"Column index must be integer or None. Got {type(col2)} for column 2 in pair {pair_idx}")
                if col2 < 0 or col2 >= self.X.shape[1]:
                    raise ValueError(f"Column index {col2} out of bounds for matrix with {self.X.shape[1]} columns")
                pair_rows.update(self.X.getcol(col2).nonzero()[0])
            
            # If both columns in a pair are None, skip this pair
            if col1 is None and col2 is None:
                continue
                
            # Create mask for current pair
            current_mask = np.zeros(self.X.shape[0], dtype=bool)
            current_mask[list(pair_rows)] = True
            
            # Update overall mask (AND condition)
            row_mask &= ~current_mask

        # Add time bounds filter
        # print(self.df.head())
        # print(self.date_range)
        if self.date_range:
            row_mask &= (self.df['Date'] >= self.date_range[0]) & (self.df['Date'] <= self.date_range[1])
            # if self.date_range[1]:
            #     row_mask 
        
        # Return row indices that satisfy all conditions
        # print(row_mask)
        self.valid_rows = np.where(row_mask)[0]
        self.valid_wr_rows = np.where(row_mask & ~self.df['Invalid_WR'])[0]
        # print(self.valid_wr_rows)

    @param.depends('valid_rows')
    def get_selection_info(self):
        return pn.Row(
            pn.pane.Markdown(
                f'You have selected {self.valid_rows.shape[0]} decks, {self.valid_wr_rows.shape[0]} of which have valid win rate information.',
                sizing_mode='stretch_width',
            ),
            pn.widgets.TooltipIcon(
                value="League data and other sources only show decks with 100% winrate, so they can't be included in win rate calculations. They still contribute to aggregation info.",
                max_width=10
            ),
        )

    @param.depends('selected_cards', 'valid_rows')
    def get_deck_view(self):
        valid_cards = np.unique(self.X[self.valid_rows].nonzero()[1])
        # if valid_cards.shape[0] > 500:
        #     return pn.pane.Markdown(
        #         f'''Too many cards to display deck aggregation. Make a more restrictive filter.
        #         Current cards: {valid_cards.shape[0]}, Max cards: 500
        #         '''
        #     )
        
        # Work out how many of each card is played in aggregate.
        #
        counts_df = pd.DataFrame(
            sparse_column_value_counts(
                self.X[self.valid_rows][:, valid_cards]
            )
        ).fillna(0)

        # Index properly by card name.
        #
        idx_card_map = {v: k for k, v in self.feature_names.items()}
        counts_df.index = [idx_card_map.get(c) for c in valid_cards]
        counts_df.index.name = 'Card'

        # Handle for when we have more than 4 of a card.
        # We should be able to aggregate to 4+ without losing value.
        # Mono color standard or limited decks are the only real issue here.
        #
        if any(counts_df.columns>4):
            counts_df['4+'] = np.nansum(counts_df[[col for col in counts_df.columns if col>=4]], axis=1)
            counts_df = counts_df.rename(columns={0:'0',1:'1',2:'2',3:'3'})
            col_list = ['0','1','2','3','4+']
            
        else:
            counts_df = counts_df.rename(columns={0:'0',1:'1',2:'2',3:'3',4:'4'})
            col_list = ['0','1','2','3','4']

        counts_df = counts_df[col_list]
        counts_df.fillna(0)

        # Split into main/sb.
        #
        mb_counts_df = counts_df.loc[
            [i for i in counts_df.index if not i.endswith('_SB')]
        ].sort_values(
            col_list
        )

        sb_counts_df = counts_df.loc[
            [i for i in counts_df.index if i.endswith('_SB')]
        ].sort_values(
            col_list
        )

        # Remove the _SB suffix
        #
        sb_counts_df.index = [c[:-3] for c in sb_counts_df.index]
        sb_counts_df.index.name = 'Card'

        # Preprocess DataFrame to apply HTML formatter
        #
        for col in col_list:
            mb_counts_df[col] = mb_counts_df[col].apply(vertical_bar_html)
            sb_counts_df[col] = sb_counts_df[col].apply(vertical_bar_html)

        # Create tabulator with HTML formatter
        #
        formatters = {
            col: {'type': 'html'} for col in col_list
        }

        mb_table = pn.widgets.Tabulator(
            mb_counts_df, formatters=formatters, pagination='local',
        )
        sb_table = pn.widgets.Tabulator(
            sb_counts_df, formatters=formatters, pagination='local',
        )
    
        return pn.Row(
            pn.Column(
                pn.pane.HTML('<h3>Main</h3>'),
                mb_table,
            ), 
            pn.Column(
                pn.pane.HTML('<h3>Sideboard</h3>'),
                sb_table,
            ),
        )
     
    @param.depends('selected_card', 'show_correlation', 'valid_rows')
    def get_card_analysis(self):
        if not self.selected_card:
            return pn.pane.Markdown("Select a card and enable correlation view to see analysis")
            
        # Calculate correlation matrix for main/sideboard copies
        mb_idx = self.feature_names.get(self.selected_card)
        sb_idx = self.feature_names.get(f"{self.selected_card}_SB")
        
        if (mb_idx is None) and (sb_idx is None):
            return pn.pane.Markdown("Card not found in dataset")
        
        if mb_idx is None:
            mb_copies = [np.nan]
            _, _, sb_copies = sparse.find(self.X[self.valid_rows][:, sb_idx])
            n_decks = sb_copies.shape[0]
        elif sb_idx is None:
            sb_copies = [np.nan]
            _, _, mb_copies = sparse.find(self.X[self.valid_rows][:, mb_idx])
            n_decks = mb_copies.shape[0]
        else:
            mb_d, _ = self.X[self.valid_rows][:, mb_idx].nonzero()
            sb_d, _ = self.X[self.valid_rows][:, sb_idx].nonzero()
            d = set(np.concatenate([mb_d, sb_d]))
            mb_copies = self.X[self.valid_rows][list(d), mb_idx].toarray().flatten()
            sb_copies = self.X[self.valid_rows][list(d), sb_idx].toarray().flatten()
            n_decks = len(d)

        bins = np.arange(-0.5, np.nanmax([np.nanmax(mb_copies), np.nanmax(sb_copies), 5]), 1)
        mb_y, _ = np.histogram(mb_copies, bins, density=True)
        sb_y, _ = np.histogram(sb_copies, bins, density=True)

        return hv.Bars(
            pd.DataFrame({
                'Frequency': np.concatenate([mb_y, sb_y]),
                'Qtty': [0,1,2,3,4,0,1,2,3,4],
                'Board': ['M']*5 + ['SB'] * 5,
                # 'B': ['Main'] * 5 + ['Sideboard'] * 5
            }),
            kdims=['Qtty', 'Board'],
        ).opts(
            width=400,
            height=400,
            # multi_level=False,
            title=f"Qtty Frequency",
        )
    
    @param.depends('selected_card', 'valid_wr_rows')
    def get_winrate_analysis(self):
        """Todo: Error bars, total."""
        if not self.selected_card:
            return pn.pane.Markdown("Select a card to see win rate analysis")
            
        # Calculate win rates by copy count
        
        if not self.selected_card in self.feature_names:
            return pn.pane.Markdown("Card not found in dataset")
        
        plots = list()
        
        mb_idx = self.feature_names.get(self.selected_card)
        if mb_idx is not None:    
            copy_counts = self.X[self.valid_wr_rows][:, mb_idx].toarray()

            # print(self.df.loc[self.valid_wr_rows,['Wins']].value_counts(), copy_counts)
            
            mb_win_rates = []
            for i in range(5):  # 0-4 copies
                mask = copy_counts == i
                wins = self.df.loc[self.valid_wr_rows].reset_index().loc[mask.ravel(), 'Wins'].sum()
                total = wins + self.df.loc[self.valid_wr_rows].reset_index().loc[mask.ravel(), 'Losses'].sum()
                if total:
                    ci = binomtest(k=int(wins), n=int(total)).proportion_ci()
                    winrate = wins/total if total else np.nan
                    mb_win_rates.append({
                        'copies': i-0.1, 
                        'winrate': winrate,
                        'errmin': winrate - ci.low,
                        'errmax': ci.high - winrate,
                    })
                else:
                    # For completeness
                    #
                    mb_win_rates.append({
                        'copies': i-0.1, 
                        'winrate': np.nan,
                        'errmin': np.nan,
                        'errmax': np.nan,
                    })
            
            plots.append(hv.Scatter(
                mb_win_rates, 'copies', 'winrate', label='Main',
            ).opts(size=7,))
            plots.append(hv.ErrorBars(
                mb_win_rates, 'copies', vdims=['winrate', 'errmin', 'errmax'],
            ))

        sb_idx = self.feature_names.get(f'{self.selected_card}_SB')
        if sb_idx is not None:    
            copy_counts = self.X[self.valid_wr_rows][:, sb_idx].toarray()
            
            sb_win_rates = []
            for i in range(5):  # 0-4 copies
                mask = copy_counts == i
                wins = self.df.loc[self.valid_wr_rows].reset_index().loc[mask.ravel(), 'Wins'].sum()
                total = wins + self.df.loc[self.valid_wr_rows].reset_index().loc[mask.ravel(), 'Losses'].sum()

                if total:
                    ci = binomtest(k=int(wins), n=int(total)).proportion_ci()
                    winrate = wins/total if total else np.nan
                    sb_win_rates.append({
                        'copies': i+0.1, 
                        'winrate': winrate,
                        'errmin': winrate - ci.low,
                        'errmax': ci.high - winrate,
                    })
                else:
                    # For completeness
                    #
                    sb_win_rates.append({
                        'copies': i+0.1, 
                        'winrate': np.nan,
                        'errmin': np.nan,
                        'errmax': np.nan,
                    })
            
            plots.append(hv.Scatter(
                sb_win_rates, 'copies', 'winrate', label='Sideboard',
            ).opts(size=7,))
            plots.append(hv.ErrorBars(
                sb_win_rates, 'copies', vdims=['winrate', 'errmin', 'errmax'],
            ))

        # Add helper lines.
        #
        wins = self.df.loc[self.valid_wr_rows]['Wins'].sum()
        total = wins + self.df.loc[self.valid_wr_rows]['Losses'].sum()
        wr = wins/total
        # return hv.Curve([(0.5, 0.5),(5.5, 0.5)], 'copies', label='50% wr').opts(color='k', line_dash='dotted')

        plots.extend([
            hv.Curve([(-0.5, 0.5),(4.5, 0.5)], 'copies', 'winrate', label='50% wr').opts(color='k', line_dash='dotted'),
            hv.Curve([(-0.5, wr),(4.5, wr)], 'copies', 'winrate', label='Deck average').opts(color='k', line_dash='dashed')
        ])
                
        # Create line plot using HoloViews
        win_rate_plot = hv.Overlay(plots).opts(
            width=400,
            height=400,
            title=f"Win Rate by Copy Count",
            ylabel='Win Rate',
            xlabel='Number of Copies',
            xlim=(-0.5, 4.5),
            ylim=(-0.1, 1.1),
        )
        
        return win_rate_plot


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

# Create the dashboard
def create_dashboard(df, X, vocabulary):
    analyzer = MTGAnalyzer(df, X, vocabulary)
    
    # Create card selection widget
    card_select = pn.widgets.MultiChoice(
        name='Required Cards',
        options=analyzer.card_options,
        value=[],
        placeholder='Search for cards...',
        sizing_mode='stretch_width'
    )

    # Create card selection widget
    card_exclude = pn.widgets.MultiChoice(
        name='Excluded Cards',
        options=analyzer.card_options,
        value=[],
        placeholder='Search for cards...',
        sizing_mode='stretch_width'
    )
    
    # Create date range selector
    date_range = pn.widgets.DateRangeSlider(
        name='Date Range',
        start=df['Date'].min(),
        end=df['Date'].max(),
        value=(df['Date'].max() - pd.Timedelta(weeks=3*4), df['Date'].max()),
        sizing_mode='stretch_width'
    )
    
    # Create card analysis widgets
    card_analysis = pn.widgets.Select(
        name='Analyze Card',
        options=[''] + analyzer.card_options,
        # value=[''],##
        sizing_mode='stretch_width'
    )
    
    # Link widgets to analyzer parameters
    card_select.link(analyzer, value='selected_cards')
    card_exclude.link(analyzer, value='excluded_cards')
    card_analysis.link(analyzer, value='selected_card')
    date_range.link(analyzer, value='date_range')

    description = pn.pane.HTML(
        '''
        Urza's Research Desk brought to you by me, <a target="_blank" rel="noopener noreferrer" href="https://bsky.app/profile/arckaynine.bsky.social">ArcKayNine</a>.<br>
        Data comes courtesy of the excellent work done by <a target="_blank" rel="noopener noreferrer" href="https://github.com/Badaro/MTGODecklistCache">Badaro</a>.<br>      
        For more of my work, check out my blog, <a target="_blank" rel="noopener noreferrer" href="https://compulsiveresearchmtg.blogspot.com">CompulsiveResearchMtg</a> or the exploits of my team, <a href="https://bsky.app/profile/busstop-mtg.bsky.social">Team Bus Stop</a>.<br>
        If you find this useful, valuable, or interesting, consider supporting further work via my <a target="_blank" rel="noopener noreferrer" href="https://ko-fi.com/arckaynine">Ko-fi</a>.<br>
        ''',
        # width=150,
    )
    
    # Create layout groups
    controls = pn.Column(
        pn.pane.Markdown("## MTG Deck Analysis"),
        pn.pane.Markdown("To filter down the decks you're looking at, select cards that are required in the 75, and cards that cannot be in the 75."),
        card_select,
        card_exclude,
        date_range,
        analyzer.get_selection_info,
        description,
        sizing_mode='stretch_width'
    )
    
    aggregate_view = pn.Column(
        analyzer.get_deck_view,
        sizing_mode='stretch_both',
        name='Aggregate Deck Analysis'
    )
    
    analysis_view = pn.Column(
        card_analysis,
        pn.Row(
            analyzer.get_card_analysis,
            analyzer.get_winrate_analysis,
        ),
        sizing_mode='stretch_both',
        name='Card Performance Analysis'
    )
    
    # Create template
    template = pn.template.FastListTemplate(
        title="Urza's Research Desk",
        sidebar=[controls],
        main=[
            pn.Tabs(
                aggregate_view,
                analysis_view,
                # TODO
                # Temporal analysis (moving average population + wr)
                sizing_mode='stretch_width'
            )
        ],
    )
    
    return template


# if __name__ == '__main__':
df, X, vocabulary = load_data()
dashboard = create_dashboard(df, X, vocabulary)
dashboard.servable()