#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px


# In[161]:


df = pd.read_csv("E:\\143\\indicators.csv.zip")


# In[162]:


oecd = ['United States', 'United Kingdom', 'Australia', 'Canada' , 'France', 'Germany', 'Greece', 'Japan',         'Sweden', 'Switzerland']

asian = ['India', 'China', 'Indonesia', 'Pakistan', 'Bangladesh', 'Vietnam', 'Korea, Rep.', 'Nepal', 'Sri Lanka']

arabic = ['Egypt, Arab Rep.', 'Iraq', 'Saudi Arabia', 'United Arab Emirates', 'Lebanon', 'Saudi Arabia', 'Turkey']

african = ['Nigeria', 'Kenya', 'Uganda', 'Sudan', 'Ethiopia', 'Botswana', 'Namibia', 'Rwanda', 'Zimbabwe']

latin = ['Argentina', 'Brazil', 'Colombia', 'Mexico', 'Venezuela, RB']

region = dict(list(zip(oecd, ['OECD']*len(oecd))) + list(zip(asian, ['Asian']*len(asian))) + list(zip(arabic, ['Arab']*len(arabic)))           + list(zip(african, ['Sub-Saharan Africa']*len(african))) + list(zip(latin, ['Latin-America']*len(african))))
        

countries =  oecd + asian + arabic + african +latin

all_countries = set(df['CountryName'])


# In[164]:


Indicator_array =  df[['IndicatorName','IndicatorCode']].drop_duplicates().values
modified_indicators = []
unique_indicator_codes = []
for ele in Indicator_array:
    indicator = ele[0]
    indicator_code = ele[1].strip()
    if indicator_code not in unique_indicator_codes:
        # delete , ( ) from the IndicatorNames
        new_indicator = re.sub('[,()]',"",indicator).lower()
        # replace - with "to" and make all words into lower case
        new_indicator = re.sub('-'," to ",new_indicator).lower()
        modified_indicators.append([new_indicator,indicator_code])
        unique_indicator_codes.append(indicator_code)

Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])
Indicators = Indicators.drop_duplicates()
print(Indicators.shape)

key_word_dict = {}
key_word_dict['Demography'] = ['parliament', 'population','birth','death','fertility','mortality','expectancy']
key_word_dict['Food'] = ['food','grain','nutrition','calories']
key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
key_word_dict['Health'] = ['survival','health','desease','hospital','mortality','doctor', 'expectancy']
key_word_dict['Economy'] = ['gini','income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy']
key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']
key_word_dict['Rural'] = ['rural','village']
key_word_dict['Urban'] = ['urban','city']


def print_indicators(feature):

    for indicator_ele in Indicators.values:
        for ele in key_word_dict[feature]:
            word_list = indicator_ele[0].split()
            if ele in word_list or ele+'s' in word_list:
                print(indicator_ele)
                break
            


# In[165]:


df.set_index('CountryName', inplace = True)
df_short = df.copy()
remaining_countries = all_countries - set(countries)
df_short.drop(index = remaining_countries, inplace = True)

df_short.reset_index(drop = False, inplace = True)

df_short = df_short.drop(columns = ['CountryCode', 'IndicatorName'])
df_pivot = df_short.pivot_table( values = 'Value' , columns= ['IndicatorCode'], index = ['CountryName', 'Year']).reset_index()


# In[166]:


##expectation and fertility rate DataFrame
df_expect_fert = df_pivot[['CountryName', 'Year', 'SP.DYN.LE00.IN', 'SP.DYN.TFRT.IN', 'SP.POP.TOTL']]

df_expect_fert.rename(columns =                       {'SP.DYN.LE00.IN': 'life_expectancy', 'SP.DYN.TFRT.IN': 'fertility', 'SP.POP.TOTL': 'tot_pop'},                     inplace = True)

df_expect_fert = df_expect_fert[~((df_expect_fert['Year'] == 2015) | (df_expect_fert['Year'] == 2014 ))]
df_expect_fert['Group'] = df_expect_fert['CountryName'].apply(lambda x: region[x])

######plot#########
fig = px.scatter(df_expect_fert,x='fertility', y='life_expectancy',animation_frame='Year',  
 animation_group='CountryName',size='tot_pop',  
 color='Group',
 hover_name='CountryName', #log_x=True, 
    size_max=60, range_y = [20,100] ,range_x=[-1,10]
 )
# Tune marker appearance and layout
fig.update_traces(mode='markers', marker=dict(sizemode='area',
 ))
fig.update_layout(
 title='Life Expectancy v. Fertility Rates (1960-2013)',
 xaxis=dict(
 title= 'Fertility Rate',
 gridcolor='white',
#  type='linear',
 gridwidth=2,
 ),
 yaxis=dict(
 title='Life Expectancy (Years)',
 gridcolor='white',
 gridwidth=2,
 ),
 paper_bgcolor='rgb(243, 243, 243)',
 plot_bgcolor='rgb(243, 243, 243)',
)
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
fig.show()
fig.write_html("life_expect.html")


# In[167]:


###mortality rate and gdp per capita DF
df_mort_gdp = df_pivot[['CountryName', 'Year', 'SH.DYN.MORT', 'NY.GDP.PCAP.KD', 'SP.POP.TOTL']]

df_mort_gdp.rename(columns =                       {'SH.DYN.MORT': 'child_mortality', 'NY.GDP.PCAP.KD': 'GDP_per', 'SP.POP.TOTL': 'tot_pop'},                     inplace = True)

df_mort_gdp = df_mort_gdp[~((df_mort_gdp['Year'] == 2015) | (df_mort_gdp['Year'] <1984))]
df_mort_gdp['Group'] = df_mort_gdp['CountryName'].apply(lambda x: region[x])

df_mort_gdp = df_mort_gdp[~df_mort_gdp['GDP_per'].isnull()]


####plot################
fig1 = px.scatter(df_mort_gdp,x='GDP_per', y='child_mortality',animation_frame='Year',  
 animation_group='CountryName',size='tot_pop',  
 color='Group',
 hover_name='CountryName', log_x=True, 
    size_max=60, range_y = [-1,300] ,range_x=[10,100000]
 )
# Tune marker appearance and layout
fig1.update_traces(mode='markers', marker=dict(sizemode='area',
 ))
fig1.update_layout(
 title='Child Mortality per 1000 vs GDP per capita (1984-2014)',
 xaxis=dict(
 title= 'GDP per capita, USD',
 gridcolor='white',
#  type='linear',
 gridwidth=2,
 ),
 yaxis=dict(
 title= 'Child Mortality Rate per 1000',
 gridcolor='white',
 gridwidth=2,
 ),
 paper_bgcolor='rgb(243, 243, 243)',
 plot_bgcolor='rgb(243, 243, 243)',
)
fig1.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
fig1.show()
fig1.write_html("mort.html")


# In[168]:


######income share of richest 10%
import seaborn as sns
df_rich = pd.read_csv("E:\\143\\income_share_of_richest_10percent.csv")
df_rich = df_rich[['country', '2016']].sort_values(by = '2016', ascending = False).dropna()

country == [ 'United States', 'Finland', 'Greece', 'Uganda', 'Bangladesh''China', 'Brazil', 'Sweden', 'Norway', 'Colombia', 'Germany', 'United Kingdom']

income_share = df_rich.query("country == [ 'United States', 'Finland', 'Greece', 'Uganda', 'Bangladesh''China', 'Brazil', 'Sweden', 'Norway', 'Colombia', 'Germany', 'United Kingdom']")
df_rich = df_rich.iloc[:10, :]


graph1 = sns.barplot(x = "2016", y = "country",  palette = 'ch:s=.25,rot=-.25', data =income_share )
plt.xlabel('Percentage of Wealth Captured', fontsize = 14)
plt.ylabel('Country', fontsize = 14)
plt.title('Income share of richest 10%', fontsize = 14)
plt.xlim(0, 50)
plt.show()


# In[ ]:




