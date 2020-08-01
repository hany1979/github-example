#!/usr/bin/env python
# coding: utf-8

# # Peer-graded Assignment Capstone-Open a Italian Restaurant in NYC:

# ## A. Introduction & Business Problem and Describe the Data:

# ### A.1. Introduction / Business Problem: 

# <font size ="3">
# The City of New York, is the most populous city in the United States. It is diverse and is the financial capital of USA. It is multicultural with 19.45 million people. New York City welcomed a record of 65.2 million visitors, comprising 51.6 million domestic and 13.5 million international visitors in 2019, It provides lot of business opportunities  and business friendly environment. one of Largest cities by population in New York is Brooklyn, this city has a lot of immigrants from all over the world with many cultural diversity which allowing a huge types of cuisines one can offer in the restaurant business.
# 
# Demographics of Brooklyn:- 
#     
# | Ancestry  |  Number  | % of total population  |
# |:--------:|:--------:|:----------------------:|
# | Italian  | 157,068  | 6.1%                   |
# | Irish    | 100,923  | 3.9%                   |
# | Russian  | 88,766   | 3.5%                   |
# | Polish   | 71,099   | 2.8%                   |
# | German   | 53,188   | 2.1%                   |
# | English  | 36,174   | 1.4%                   |
#     
# 
# As it is highly developed city so cost of doing business is also one of the highest. Thus, any new business venture or expansion needs to be analyzed carefully. The insights derived from analysis will give good understanding of the business environment which help in strategically targeting the market. This will help in reduction of risk and the return on investment will be reasonable. 
# 
# Our analysis will be covered the following factors:
# * The best location to start the business.
# * The types of our target Customers.
# * defined our Potential competitors?
# </font>
# 
# 

# ### A.2. Import required Libraries:

# <font size = '3'>To Identify the best location to start the business in Brooklyn, I need to find out the number of our competitors in each neighborhoods in Brooklyn .
# 
# I used NYC OpenData to import the data of "Neighborhoods in New York City
#     
# https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas-NTA-/cpf4-rkhq
# 
# In Brooklyn, there is 576 Italian restaurants are currently running. 
# 
# I used Forsquare API to get the most common venues of given Borough of Brooklyn and the Italian restuarants Id.
#     
# https://foursquare.com/ 
#     
#  I used Google API to find their geographic coordinates based on their postal code addresses.
# 
# </font>
# 

# ## B. Methodology:

# ### B.1. Import required Libraries:

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files
from pprint import pprint # data pretty printer

import requests # library to handle requests
from bs4 import BeautifulSoup  # library to handle web scraping

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 

# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

import folium # map rendering library

import matplotlib.cm as cm # Matplotlib and associated plotting modules
import matplotlib.colors as colors # Matplotlib and associated plotting modules

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

from collections import Counter # count occurrences 

from sklearn.cluster import KMeans # import k-means from clustering stage

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library

print('All required Libraries imported')


# ### B.2. Read and Explore New York City Dataset:

# ### - Read The Data:

# In[2]:


# Read Data 
df = pd.read_csv("NYC_Data.csv")
df.head()


# ### - Clean The Data:

# In[3]:


# drop unnecessary Columns 
NYC_df = df.drop(columns = ['OBJECTID', 'Stacked', 'AnnoLine1', 'AnnoLine2', 'AnnoLine3', 'AnnoAngle'])
NYC_df = df[['Borough', 'Name', 'Latitude', 'Longitude']]
NYC_df.head()


# In[4]:


# Rename 'Name' column to 'Neighborhood'
NYC_df = NYC_df.rename(columns={'Name': 'Neighborhood'})
NYC_df.reset_index(drop=True, inplace=True) # reset index 
NYC_df.head()


# In[5]:


# returen the numbers of columns and rows 
NYC_df.shape


# ### B.3. Create a map of New York with neighborhoods.

# In[6]:


address = 'New York City, NY'
location = None

# define an instance of the geocoder -> ny_explorer
while location == None:
    try:
        geolocator = Nominatim(user_agent="ny_explorer")
        location = geolocator.geocode(address)
        latitude = location.latitude
        longitude = location.longitude
    except:
        pass
print("latitude: ", latitude, " & longitude:" ,longitude)


# In[7]:


# Create Brooklenn dataframe
brooklyn_data = NYC_df[NYC_df['Borough'] == 'Brooklyn'].reset_index(drop=True)
brooklyn_data.head()


# In[8]:


# create map of Brooklyn, New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(brooklyn_data['Latitude'], brooklyn_data['Longitude'], brooklyn_data['Borough'], brooklyn_data['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#d6add5',
        fill_opacity=0.7,
        parse_html=False).add_to(map_newyork)  
    
map_newyork


# ### B.4. Define Foursquare Credentials and Version:

# In[9]:


# Define Foursquare Credentials and Version
CLIENT_ID = 'ON5YVHHLDSVO0R0FEBOTOR3VZRAC1WXSOGL4CVP5C0DGLCUZ' 
CLIENT_SECRET = '5BDHPM1UXLUJSG2C1AFGSTAB55CUTXGIWDTL1330QCBXDNCJ' 
VERSION = '20200726'
LIMIT = 5000
radius = 5000 
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ### Let's  relevant information for each nearby venue:

# In[10]:


import urllib
def getNearbyVenues(names, latitudes, longitudes, radius=5000, categoryIds=''):
    try:
        venues_list=[]
        for name, lat, lng in zip(names, latitudes, longitudes):
            #print(name)

            # create the API request URL
            url = 'https://api.foursquare.com/v2/venues/search?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, VERSION, lat, lng, radius, LIMIT)

            if (categoryIds != ''):
                url = url + '&categoryId={}'
                url = url.format(categoryIds)

            # make the GET request
            response = requests.get(url).json()
            results = response["response"]['venues']

            # return only relevant information for each nearby venue
            for v in results:
                success = False
                try:
                    category = v['categories'][0]['name']
                    success = True
                except:
                    pass

                if success:
                    venues_list.append([(
                        name, 
                        lat, 
                        lng, 
                        v['name'], 
                        v['location']['lat'], 
                        v['location']['lng'],
                        v['categories'][0]['name']
                    )])

        nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
        nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude',  
                  'Venue Category']
    
    except:
        print(url)
        print(response)
        print(results)
        print(nearby_venues)

    return(nearby_venues)


# In[11]:


# run the function on each neighborhood and create a new dataframe
brooklyn_venues = getNearbyVenues(names = brooklyn_data['Neighborhood'],
                                 latitudes = brooklyn_data['Latitude'],
                                 longitudes = brooklyn_data['Longitude']
                                 )


# In[12]:


# filter data of Venue Category 
Brooklyn_food_venues = brooklyn_venues[brooklyn_venues['Venue Category'].str.contains(r'Food(?!$)', r'Restaurant(?!$)')]
Brooklyn_food_venues.reset_index(drop=True, inplace=True)
Brooklyn_food_venues.head()


# In[13]:


Brooklyn_food_venues.shape


# In[14]:


# Let's check how many venues were returned for each neighborhood
Brooklyn_food_venues.groupby('Neighborhood').count()


# In[15]:


# let's chech the uniques categories in neighborhood
print('There are {} uniques categories.'.format(len(Brooklyn_food_venues['Venue Category'].unique())))
Brooklyn_food_venues.groupby('Venue Category')['Venue Category'].count().sort_values(ascending=False)


# In[16]:


# one hot encoding
brooklyn_onehot = pd.get_dummies(Brooklyn_food_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
brooklyn_onehot['Neighborhood'] = Brooklyn_food_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [brooklyn_onehot.columns[-1]] + list(brooklyn_onehot.columns[:-1])
brooklyn_onehot = brooklyn_onehot[fixed_columns]

brooklyn_onehot.head()


# In[17]:


venue_counts = brooklyn_onehot.groupby('Neighborhood').sum()
venue_counts.head(5)


# In[18]:


#Let's find out the top 10 food categories in Brooklyn
venue_counts_described = brooklyn_onehot.describe().transpose()
venue_top10 = venue_counts_described.sort_values('max', ascending=False)[0:10]
venue_top10


# In[19]:


venue_top10_list = venue_top10.index.values.tolist()


# In[20]:


# plot the top 10 food categories in the Neighborhood 
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes =plt.subplots(5, 2, figsize=(20,20), sharex=True)
axes = axes.flatten()

for ax, category in zip(axes, venue_top10_list):
    data = venue_counts[[category]].sort_values([category], ascending=False)[0:10]
    pal = sns.color_palette("Reds", len(data))
    sns.barplot(x=category, y=data.index, data=data, ax=ax, palette=np.array(pal[::-1]))

plt.tight_layout()
plt.show();


# In[21]:


# Let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
brooklyn_grouped = brooklyn_onehot.groupby('Neighborhood').mean().reset_index()
brooklyn_grouped.head()


# In[22]:


brooklyn_grouped.shape


# In[23]:


# import matplotlib and matplotlib.pylot
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')

brooklyn_grouped.plot(kind='hist', figsize=(12, 8))
plt.title('Top Venue in Neighborhood') # add a title to the histogram
plt.show()


# In[24]:


# Let's print each neighborhood along with the top 5 most common venues
num_top_venues = 10

for hood in brooklyn_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = brooklyn_grouped[brooklyn_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[25]:


# Let's put that information into a pandas dataframe

# Function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[26]:


# create the new dataframe and display the top 5 venues for each neighborhood
num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = brooklyn_grouped['Neighborhood']

for ind in np.arange(brooklyn_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(brooklyn_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[27]:


# get Italian Restaurant id https://developer.foursquare.com/docs/resources/categories 
#Italian Restaurant = 4bf58dd8d48988d110941735
NYC_df = NYC_df[NYC_df['Borough'] == 'Brooklyn'].reset_index(drop=True)
newyork_venues_italian = getNearbyVenues(names=NYC_df['Neighborhood'], latitudes=NYC_df['Latitude'], longitudes=NYC_df['Longitude'], radius=1000, categoryIds='4bf58dd8d48988d110941735')
newyork_venues_italian.head()


# In[28]:


newyork_venues_italian.shape


# In[29]:


def addToMap(df, color, existingMap):
    for lat, lng, local, venue, venueCat in zip(df['Venue Latitude'], df['Venue Longitude'], df['Neighborhood'], df['Venue'], df['Venue Category']):
        label = '{} ({}) - {}'.format(venue, venueCat, local)
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='Blue',
            fill=True,
            fill_color='#d6add5',
            fill_opacity=0.7).add_to(existingMap)


# In[30]:


map_newyork_italian = folium.Map(location=[latitude, longitude], zoom_start=9)
addToMap(newyork_venues_italian, 'blue', map_newyork_italian)

map_newyork_italian


# ### C. Analyzing each neighborhood

# In[31]:


# one hot encoding
brooklyn_onehot = pd.get_dummies(newyork_venues_italian[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
brooklyn_onehot['Neighborhood'] = newyork_venues_italian['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [brooklyn_onehot.columns[-1]] + list(brooklyn_onehot.columns[:-1])
brooklyn_onehot = brooklyn_onehot[fixed_columns]

brooklyn_onehot.head()


# In[32]:


# Checking the size of the DataFrame
brooklyn_onehot.shape


# In[33]:


# Let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
brooklyn_grouped = brooklyn_onehot.groupby('Neighborhood').mean().reset_index()
brooklyn_grouped


# In[34]:


# Confirm the new size
brooklyn_grouped.shape


# In[35]:


# Let's print each neighborhood along with the top 5 most common venues
num_top_venues = 5

for hood in brooklyn_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = brooklyn_grouped[brooklyn_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[36]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[37]:


# create the new dataframe and display the top 10 venues for each neighborhood
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = brooklyn_grouped['Neighborhood']

for ind in np.arange(brooklyn_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(brooklyn_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### D. Cluster Neighborhoods

# ### Run k-means to count Neighborhoods for each cluster label for variable cluster size

# In[38]:


brooklyn_grouped_clustering = brooklyn_grouped.drop('Neighborhood', 1)


# In[39]:


from sklearn.metrics import silhouette_score

sil = []
K_sil = range(2,20)
# minimum 2 clusters required, to define dissimilarity
for k in K_sil:
    print(k, end=' ')
    kmeans = KMeans(n_clusters = k).fit(brooklyn_grouped_clustering)
    labels = kmeans.labels_
    sil.append(silhouette_score(brooklyn_grouped_clustering, labels, metric = 'euclidean'))


# In[40]:


plt.plot(K_sil, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette_score')
plt.title('Silhouette Method For Optimal k')
plt.show()


# There is a peak at k = 2, k = 4 and k = 8. Two and four clusters will give a very broad classification of the venues.

# In[41]:


# set number of clusters
kclusters = 8

# run k-means clustering
kmeans = KMeans(init="k-means++", n_clusters=kclusters,  random_state=0).fit(brooklyn_grouped_clustering)

print(Counter(kmeans.labels_))


# In[42]:


# add clustering labels
try:
    neighborhoods_venues_sorted.drop('Cluster Labels', axis=1)
except:
    neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[43]:


neighborhoods_venues_sorted.shape


# In[44]:


neighborhoods_venues_sorted.head(5)


# In[45]:


# merge neighborhoods_venues_sorted with nyc_data to add latitude/longitude for each neighborhood
brooklyn_merged = neighborhoods_venues_sorted.join(NYC_df.set_index('Neighborhood'), on='Neighborhood')
brooklyn_merged.head()


# In[46]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
colors_array = cm.rainbow(np.linspace(0, 1, kclusters))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(brooklyn_merged['Latitude'], brooklyn_merged['Longitude'], brooklyn_merged['Neighborhood'], brooklyn_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### D.1. Analyze clusters - Results:

# In[47]:


required_column_indices = [2,3,7]
required_column = [list(brooklyn_merged.columns.values)[i] for i in required_column_indices]
required_column_indices = [2,3,7]


# #### D.1.1. Cluster 1, Label = 0

# In[48]:


cluster_0 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 0, brooklyn_merged.columns[1:12]]
cluster_0.head(5)


# In[49]:


for col in required_column:
    print(cluster_0[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.2.  Cluster 1, Label = 1

# In[50]:


cluster_1 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 1, brooklyn_merged.columns[1:12]]
cluster_1.head(5)


# In[51]:


for col in required_column:
    print(cluster_1[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.3.  Cluster 2, Label = 2

# In[52]:


cluster_2 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 2, brooklyn_merged.columns[1:12]]
cluster_2.head(5)


# In[53]:


for col in required_column:
    print(cluster_2[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.4. Cluster 3, Label = 3

# In[54]:


cluster_3 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 3, brooklyn_merged.columns[1:12]]
cluster_3.head(5)


# In[55]:


for col in required_column:
    print(cluster_3[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.5.  Cluster 4, Label = 4

# In[56]:


cluster_4 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 4, brooklyn_merged.columns[1:12]]
cluster_4.head(5)


# In[57]:


for col in required_column:
    print(cluster_4[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.6.  Cluster 5, Label = 5:

# In[58]:


cluster_5 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 5, brooklyn_merged.columns[1:12]]
cluster_5.head(5)


# In[59]:


for col in required_column:
    print(cluster_5[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.7.  Cluster 6, Label = 6:

# In[60]:


cluster_6 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 6, brooklyn_merged.columns[1:12]]
cluster_6.head(5)


# In[61]:


for col in required_column:
    print(cluster_6[col].value_counts(ascending = False))
    print("---------------------------------------------")


# #### D.1.8.  Cluster 7, Label = 7:

# In[62]:


cluster_7 = brooklyn_merged.loc[brooklyn_merged['Cluster Labels'] == 7, brooklyn_merged.columns[1:12]]
cluster_7.head(5)


# In[63]:


for col in required_column:
    print(cluster_7[col].value_counts(ascending = False))
    print("---------------------------------------------")

