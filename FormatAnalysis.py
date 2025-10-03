from tcg_research_desk import load_data, create_dashboard

df, X, res_df, vocabulary, oracleid_lookup, cards_data, cluster_map, clusters_id, archetype_list = load_data()
print(f'{df.shape=}, {X.shape=}, {res_df.shape=}, {len(archetype_list)=}')
dashboard = create_dashboard(df, res_df, X, vocabulary, oracleid_lookup, cards_data, cluster_map, clusters_id, archetype_list)
dashboard.servable()