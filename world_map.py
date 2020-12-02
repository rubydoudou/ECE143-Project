### This part is for Gender equality world map plot ###

import folium
import json
import pandas as pd

with open('input/world-countries.json') as data_file:
    country_geo = json.load(data_file)

data = pd.read_csv('input/Indicators.csv')

countries = data['CountryName'].unique().tolist()
indicators = data['IndicatorName'].unique().tolist()
print(len(countries))
print(len(indicators))

# Find usefule features
for i, x in enumerate(indicators):
    if 'female (%)' in x:
        print('index = %d, Indicator name is '%i + x)

# change hist_indicator to plot different indicators relative to gender equality
hist_indicator = 'Proportion of seats held by women in national parliaments (%)'

hist_year = 2011
mask1 = data['IndicatorName'].str.contains(hist_indicator,na=False, regex=False) 
mask2 = data['Year'].isin([hist_year])
# apply our mask
stage = data[mask1 & mask2]

# num of countries with this indicator
print('%d countries have this indicator'%stage.shape[0])

# generate the data I need to plot
data_to_plot = stage[['CountryCode','Value']]
data_to_plot.head()

# create map
hist_indicator = stage.iloc[0]['IndicatorName']
map0 = folium.Map(location=[100, 0], zoom_start=1.5)
map0.choropleth(geo_data=country_geo, data=data_to_plot,
             columns=['CountryCode', 'Value'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2, nan_fill_color='White',
             legend_name=hist_indicator)

# Create Folium plot
x = map0.save('plot/plot_gender.html')
# Import the Folium interactive html file
from IPython.display import IFrame
IFrame(src= 'plot/plot_gender.html', width=1000 ,height=450)

### This part is for GINI index world map plot ###
# Please restart the kernel if using jupyter notebook.

import folium
import json
import pandas as pd

with open('input/world-countries.json') as data_file:
    country_geo = json.load(data_file)
    
# read gini data
gini = pd.read_csv('input/gini.csv')
countries = pd.read_csv('input/country.csv')
countries_code = countries[['ShortName','CountryCode']]

gini.head()

# Create a country name to country code mapping to use the gini data
code_mapping = countries_code.to_dict(orient='list')

keys = code_mapping['ShortName']
values = code_mapping['CountryCode']
code_mapping_dict = dict(zip(keys,values))

data_to_plot1 = gini[['country','2019']]

# Add a new column Country Code to the dataframe
data_to_plot1['CountryCode']= data_to_plot1['country'].map(code_mapping_dict)

data_to_plot1.head()

hist_indicator1 = 'GINI index at 2019'

map1 = folium.Map(location=[100, 100], zoom_start=1)
map1.choropleth(geo_data=country_geo, data=data_to_plot1,
             columns=['CountryCode', '2019'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,nan_fill_color='White',
             legend_name=hist_indicator1)

# Create Folium plot
y = map1.save('plot/plot_GINI.html')
# Import the Folium interactive html file
from IPython.display import IFrame
IFrame(src= 'plot/plot_GINI.html', width=1000 ,height=450)




