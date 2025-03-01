import param

import numpy as np
import panel as pn
import pandas as pd
import holoviews as hv

pn.extension('tabulator', sizing_mode="stretch_width")
hv.extension('bokeh')

from scipy.sparse import find
from scipy.stats import binomtest

from Helpers import sparse_column_value_counts, vertical_bar_html

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
                value="League data and other sources only show decks with 100% winrate, so they can't be counted.",
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
            _, _, sb_copies = find(self.X[self.valid_rows][:, sb_idx])
            n_decks = sb_copies.shape[0]
        elif sb_idx is None:
            sb_copies = [np.nan]
            _, _, mb_copies = find(self.X[self.valid_rows][:, mb_idx])
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
                'Count': [0,1,2,3,4,0,1,2,3,4],
                # 'B': ['MB']*5 + ['SB'] * 5
                'B': ['Main'] * 5 + ['Sideboard'] * 5
            }),
            kdims=['Count', 'B'],
        ).opts(
            width=400,
            height=400,
            multi_level=False,
            title=f"Copy Frequency",
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
