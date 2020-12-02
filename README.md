# ECE143-Project
This is the final project for ECE143 at UCSD. 
Visual Analysis of divide between developed and developing nations on Environmental, Social and Economic Indicators
Group 7: Abhishek Yadav, Kexin Hong, Zunming Zhang, Sophia Huang


# Animated Bubble plots

The animated bubble plots can be run by changing the path for df and df_rich in bubble_and_bar_plots.py. Point the df path to World_indicators (Indicators.csv) at https://www.kaggle.com/worldbank/world-development-indicators file and df_rich path to gapminder's GINI index csv data (https://www.gapm.io/ddgini)

# World map plots

The world map plots are used to demonstrate gender equality and GINI index. To plot folium world map, download the world map json file from https://github.com/python-visualization/folium/blob/master/examples/data/world-countries.json. The data used in gender equality part involves Indicator.csv and world-countries.json. 

For GINI index, the countries with indicator GINI index in Indicator.csv file are incomplete. Therefore, use gini.csv from https://www.gapminder.org instead. Restart the kernel here. The data used in GINI index part involves Countries.csv, gini.csv and world-countries.json. 
