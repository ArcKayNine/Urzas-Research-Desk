import pandas as pd
import numpy as np

import scipy
import json
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta

import panel as pn
import param
from bokeh.palettes import Spectral6
import holoviews as hv
hv.extension('bokeh')

class MTGAnalyzer(param.Parameterized):
    selected_cards = param.List(default=[], doc="Cards required in deck")
    cluster_view = param.Boolean(default=True, doc="Show cluster view instead of card presence view")
    show_correlation = param.Boolean(default=False, doc="Show correlation heatmap for selected card")
    selected_card = param.String(default='', doc="Card to analyze in detail")
    date_range = param.DateRange(default=None, doc="Date range for analysis")
    
    def __init__(self, df, card_vectors, vocabulary, **params):
        super().__init__(**params)
        self.df = df
        self.X = card_vectors
        self.feature_names = vocabulary
        self._initialize_card_list()
        
    def _initialize_card_list(self):
        # Get unique cards from feature names, removing _SB suffix
        self.card_options = sorted(list(set(
            [name.replace('_SB', '') for name in self.feature_names]
        )))
        
    @param.depends('selected_cards', 'cluster_view', 'date_range')
    def get_deck_view(self):
        if self.cluster_view:
            return self._get_cluster_summary()
        else:
            return self._get_card_presence_view()
    
    def _get_cluster_summary(self):
        # Filter by date range if specified
        mask = np.ones(len(self.df), dtype=bool)
        if self.param.date_range.bounds:
            mask &= self.param.df['Date'] >= self.param.date_range.bounds[0]
            if self.param.date_range.bounds[1]:
                mask &= self.df['Date'] <= self.param.date_range.bounds[1]
            
        filtered_df = self.df[mask]
        
        # Group by cluster and calculate statistics
        cluster_stats = filtered_df.groupby('Cluster').agg({
            'Player': 'count',
            'Wins': ['sum', 'mean'],
            'Losses': ['sum', 'mean']
        }).round(2)
        
        # Create a tabulator widget
        return pn.widgets.Tabulator(
            cluster_stats,
            pagination='remote',
            page_size=10,
            sizing_mode='stretch_width'
        )
    
    def _get_card_presence_view(self):
        # Filter decks containing selected cards
        mask = np.ones(len(self.df), dtype=bool)
        for card in self.param.selected_cards:
            card_mask = (self.X[:, self.feature_names == card].sum(axis=1) > 0) | \
                       (self.X[:, self.feature_names == f"{card}_SB"].sum(axis=1) > 0)
            mask &= card_mask.A1
            
        filtered_df = self.df[mask]
        return pn.widgets.Tabulator(
            filtered_df[['Player', 'Wins', 'Losses', 'Date']],
            pagination='remote',
            page_size=10,
            sizing_mode='stretch_width'
        )
    
    @param.depends('selected_card', 'show_correlation')
    def get_card_analysis(self):
        if not self.param.selected_card or not self.param.show_correlation:
            return pn.pane.Markdown("Select a card and enable correlation view to see analysis")
            
        # Calculate correlation matrix for main/sideboard copies
        mb_idx = self.feature_names == self.param.selected_card
        sb_idx = self.feature_names == f"{self.selected_card}_SB"
        
        if not any(mb_idx) and not any(sb_idx):
            return pn.pane.Markdown("Card not found in dataset")
            
        deck_copies = self.X[:, mb_idx | sb_idx].toarray()
        
        # Create correlation heatmap
        corr_matrix = np.corrcoef(deck_copies.T)
        
        # Create heatmap using HoloViews
        heatmap = hv.HeatMap(
            ((range(deck_copies.shape[1]), range(deck_copies.shape[1]), corr_matrix)),
            kdims=['Main Board Copies', 'Sideboard Copies'],
            vdims=['Correlation']
        ).opts(
            width=400,
            height=400,
            cmap='RdBu_r',
            colorbar=True,
            title=f"Copy Count Correlation for {self.selected_card}"
        )
        
        return heatmap
    
    @param.depends('selected_card')
    def get_winrate_analysis(self):
        if not self.param.selected_card:
            return pn.pane.Markdown("Select a card to see win rate analysis")
            
        # Calculate win rates by copy count
        mb_idx = self.feature_names == self.param.selected_card
        if not any(mb_idx):
            return pn.pane.Markdown("Card not found in dataset")
            
        copy_counts = self.X[:, mb_idx].toarray()
        
        win_rates = []
        for i in range(5):  # 0-4 copies
            mask = copy_counts == i
            if mask.any():
                wins = self.df.loc[mask.ravel(), 'Wins'].sum()
                total = wins + self.df.loc[mask.ravel(), 'Losses'].sum()
                win_rates.append({'copies': i, 'winrate': wins/total if total else 0})
                
        # Create line plot using HoloViews
        win_rate_plot = hv.Curve(
            win_rates, 'copies', 'winrate'
        ).opts(
            width=400,
            height=300,
            title=f"Win Rate by Copy Count - {self.selected_card}",
            ylabel='Win Rate',
            xlabel='Number of Copies'
        )
        
        return win_rate_plot

# Create the dashboard
def create_dashboard(df, X, vocabulary):
    analyzer = MTGAnalyzer(df, X, vocabulary)
    
    # Create card selection widget
    card_select = pn.widgets.MultiSelect(
        name='Required Cards',
        options=analyzer.card_options,
        value=[],
        sizing_mode='stretch_width'
    )
    
    # Create date range selector
    date_range = pn.widgets.DateRangeSlider(
        name='Date Range',
        start=df['Date'].min(),
        end=df['Date'].max(),
        value=(df['Date'].min(), df['Date'].max()),
        sizing_mode='stretch_width'
    )
    
    # Create view toggle
    view_toggle = pn.widgets.Toggle(
        name='Show Cluster View',
        value=True
    )
    
    # Create card analysis widgets
    card_analysis = pn.widgets.Select(
        name='Analyze Card',
        options=[''] + analyzer.card_options,
        sizing_mode='stretch_width'
    )
    
    correlation_toggle = pn.widgets.Toggle(
        name='Show Correlation Analysis',
        value=False
    )
    
    # Link widgets to analyzer parameters
    card_select.link(analyzer, callbacks={'value': 'selected_cards'})
    view_toggle.link(analyzer, callbacks={'value': 'cluster_view'})
    card_analysis.link(analyzer, callbacks={'value': 'selected_card'})
    correlation_toggle.link(analyzer, callbacks={'value': 'show_correlation'})
    date_range.link(analyzer, callbacks={'value': 'date_range'})
    
    # Create layout
    controls = pn.Column(
        "## MTG Deck Analysis",
        card_select,
        date_range,
        view_toggle,
        card_analysis,
        correlation_toggle,
        sizing_mode='stretch_width'
    )
    
    main_view = pn.Column(
        analyzer.get_deck_view,
        sizing_mode='stretch_both'
    )
    
    analysis_view = pn.Column(
        analyzer.get_card_analysis,
        analyzer.get_winrate_analysis,
        sizing_mode='stretch_both'
    )
    
    dashboard = pn.Template(
        """
        {% extends base %}
        {% block contents %}
        <div class="container-fluid">
            <div class="row">
                <div class="col-md-3">
                    {{ controls }}
                </div>
                <div class="col-md-5">
                    {{ main_view }}
                </div>
                <div class="col-md-4">
                    {{ analysis_view }}
                </div>
            </div>
        </div>
        {% endblock %}
        """
    )
    
    dashboard.add_variable('controls', controls)
    dashboard.add_variable('main_view', main_view)
    dashboard.add_variable('analysis_view', analysis_view)
    
    return dashboard

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
        (DataFrame with deck data, sparse matrix of card counts, fitted CountVectorizer)
    """
    # Load the preprocessed data
    with open(Path(data_path) / 'deck_data.json', 'r') as f:
        data = json.load(f)
        
    # Convert to DataFrame
    df = pd.DataFrame(data['decks'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Load cluster labels
    df['Cluster'] = data['clusters']

    # Filter to recent data
    cutoff_date = pd.to_datetime('today') - pd.Timedelta(days=lookback_days) 
    df = df[df['Date'] >= cutoff_date]
    
    # Load card vectors
    X = scipy.sparse.load_npz(Path(data_path) / 'card_vectors.npz')
    
    # Load and reconstruct vectorizer
    with open(Path(data_path) / 'vectorizer.json', 'r') as f:
        vectorizer_data = json.load(f)
    
    # vectorizer = CountVectorizer()
    # vectorizer.vocabulary_ = vectorizer_data['vocabulary']
    # vectorizer.fixed_vocabulary_ = True
    
    return df, X, np.array(list(vectorizer_data['vocabulary'].keys()))

# if __name__ == '__main__':
df, X, vocabulary = load_data()
dashboard = create_dashboard(df, X, vocabulary)
dashboard.servable()