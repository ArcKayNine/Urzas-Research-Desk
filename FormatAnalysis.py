
from tcg_research_desk import load_data, create_dashboard


# if __name__ == '__main__':
df, X, res_df, vocabulary, oracleid_lookup, cards_data = load_data()
dashboard = create_dashboard(df, res_df, X, vocabulary, oracleid_lookup, cards_data)
dashboard.servable()