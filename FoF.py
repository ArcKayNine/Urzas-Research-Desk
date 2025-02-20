import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from vectorizers.transformers import InformationWeightTransformer

import json
from pathlib import Path
from datetime import datetime

import umap
import hdbscan

import thisnotthat as tnt
import panel as pn
pn.extension()
pn.extension('tabulator')

df = pd.DataFrame()

for path in Path('../MTGODecklistCache/Tournaments/').rglob('2025/*/*/*modern*.json'):
    f = open(path)
    data = json.load(f)
    
    deck_df = pd.DataFrame(data['Decks'])
    
    deck_df['Deck'] = data['Decks']
    deck_df['Tournament'] = path.name
    
    standings_df = pd.DataFrame(data['Standings'])
    if standings_df.shape[0]:
        deck_df = deck_df.join(standings_df.set_index('Player'), on='Player')
    else:
        deck_df['Wins'] = 5
        deck_df['Losses'] = 0
    
    df = pd.concat([df, deck_df.fillna(f'2024-{path.parent.parent.name}-{path.parent.name}T15:00:00Z')], ignore_index=True)

df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True)
df = df.sort_values(by='Date')

j = json.load(open('../AtomicCards.json','r'))['data']
card_list = j.keys()

def merge_analyzer(deck):
    output = list()
    for card in deck['Mainboard']:
        if card['CardName'] in card_list:
            if 'Land' not in j[card['CardName']][0]['type']:
                output += [card['CardName']] * card['Count']
        else:
            output += [card['CardName']] * card['Count']
    for card in deck['Sideboard']:
        output += [card['CardName']+'_SB'] * card['Count']
        
    return output

vectorizer = CountVectorizer(analyzer=merge_analyzer)
X = vectorizer.fit_transform(df['Deck'])

df['Plaintext'] = vectorizer.inverse_transform(X)

IWT = InformationWeightTransformer()
X_IWT = IWT.fit_transform(X)

mapper = umap.UMAP(
    metric='cosine',
    # densmap=True,
    random_state=42,
).fit(X_IWT)

clusterer = hdbscan.HDBSCAN(min_cluster_size = 100)
clusterer.fit(mapper.embedding_)

def pp_fmt(deck):
    mb_str = '  \n'.join([
        f'''{card['Count']: >2}: {card['CardName']}''' for card in deck['Mainboard']
    ])
    sb_str = '  \n'.join([
        f'''{card['Count']: >2}: {card['CardName']}''' for card in deck['Sideboard']
    ])
    
    return mb_str + '  \n----SB----  \n' + sb_str
    
df['PPrint'] = df['Deck'].apply(pp_fmt)

map_plot = tnt.BokehPlotPane(
    mapper.embedding_,
    labels=[str(l) for l in clusterer.labels_],
#     marker_size=numeric_vector,
#     hover_text=df.Archetype,
    tools='pan,wheel_zoom,lasso_select,box_select,tap,save,reset,help',
    fill_alpha=0.1,
    line_width=0,
    width=800,
)

deck_info_view = tnt.InformationPane(
    df,
    """{PPrint}
    """,
    height=800,
)
deck_info_view.link_to_plot(map_plot)

pn.serve(pn.Column(pn.Row(map_plot, deck_info_view)))

sel_1_mean = X[map_plot.selected].mean(axis=0).tolist()[0]
sel_1_std = X[map_plot.selected].toarray().std(axis=0)
print(len(map_plot.selected))

sorted([
    (m, s, n) for m, s, n in zip(
        sel_1_mean,
        sel_1_std,
        vectorizer.get_feature_names_out()
    ) if m
], key=lambda x: x[0]/x[1])[::-1]