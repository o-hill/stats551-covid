# Data

This directory stores all the relevant data needed to recreate our analysis.

## Files

<ins>nst-est2020.xlsx</ins>: 2010-2020 population estimates taken from the United States Census Bureau: https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates.html
	**Note**: This data and any interpolations made from it can be updated once the 2020 Census data is published.
	
<ins>state_populations.csv</ins>: A manually cleaned data set derived from nst-est2020.xlsx.

<ins>state_populations_clean.csv</ins>: A fully cleaned data set of state populations by year. The code used to obtain this data set is located under state_populations.ipynb.

<ins>covid_deaths.csv</ins>: A cleaned data set of Covid-19 deaths by state and month.
	**Note**: Raw data is taken from the CDC (https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Sex-Age-and-S/9bhg-hcku), but this data set gets updated weekly. To use any analysis on future dates, one will need to take this data from the CDC direclty.

- [x] data cleaning
- [ ] data population from interpolation
- [ ] automatic reading/cleaning of the CDC data
