from tcg_research_desk import process_mtg_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("format", help="Format to process", default='Modern')
    args = parser.parse_args()

    process_mtg_data(fmt=args.format)#lookback_days=30)
