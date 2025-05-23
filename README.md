# UrzasResearchDesk
A web app to display and interrogate win rates and aggregate archetypes from magic the gathering.
[See the app here!](https://arckaynine.github.io/Urzas-Research-Desk/)

## Dev Notes

### V1 Requirements
There is some major functionality/things to investigate before making this more widely available:

TODO
- Card search on mobile.
- Archetype bundles, potentially from Badaro.
- Reset all selections option.
- Plotting accessibility features (colours and fill/marker styles, not just colour).
- Bug: Unholy Annex // Ritual Chamber and Unholy Annex && Ritual Chamber both show up in the data.
- Investigate https://rtyley.github.io/bfg-repo-cleaner/ to clean up repos over time so they don't get massive.
- Show last data read time
- Loading messages

DONE (Pending testing)
- Handle for when no selected decks play 0 coppies of any card.

DONE
- Lock light/dark mode.
- Fix selection default info.
- Double check interaction between selection and individual card analysis - 0 coppies now show up.
- Lock plots in place.
- Tables can't be edited.
- Automated updates from MTGODecklistCache.
- Card images onhover.
- Fix selection updates.
- How do mixed format events (PTs) look in the data? - Just constructed rounds show up.
- Mishra's Research Desk artist credit.
- Tooltip hover not covered by tabs.
- Fix tooltip in general. - Breaks when you go straight from one to another.
- Win rate scatter legend placement.

### V2 Goals
- Temporal analysis.
- (Option to) Remove mirrors.
