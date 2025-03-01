
from MTGAnalyzer import MTGAnalyzer
from Helpers import load_data


import panel as pn
import pandas as pd
pn.extension('tabulator', throttled=True)

import holoviews as hv



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
    
    # Create layout groups
    controls = pn.Column(
        pn.pane.Markdown("## MTG Deck Analysis"),
        card_select,
        card_exclude,
        date_range,
        analyzer.get_selection_info,
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