#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install selenium')
get_ipython().system('pip install webdriver_manager')
get_ipython().system('pip install beautifulsoup4')


# In[2]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import codecs
import re
from webdriver_manager.chrome import ChromeDriverManager
import nltk
from nltk.corpus import stopwords
from selenium.webdriver.common.by import By

import nltk

# Download the 'punkt' resource
nltk.download('punkt')
nltk.download('omw-1.4')



# In[4]:


#BUSINESS

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def scrape_data_and_create_dataframe(url, seg):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    business_df_2 = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                business_df_2.append({'Genre': 'Business', 'Segment': seg, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(business_df_2)

    return df

# List of URLs and corresponding cities
url_business_pairs = [
    ("https://www.thehindu.com/business/agri-business/", "Agri-business"),
    ("https://www.thehindu.com/business/Industry/", "Industry"),
    ("https://www.thehindu.com/business/Economy/", "Economy"),
    ("https://www.thehindu.com/business/markets/", "Markets"),
    ("https://www.thehindu.com/business/budget/", "Budget")
]

# List to store all DataFrames
business_df_2 = []

for url, seg in url_business_pairs:
    df = scrape_data_and_create_dataframe(url, seg)
    business_df_2.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(business_df_2, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('Business_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)


# In[5]:


# SPORTS
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def scrape_data_and_create_dataframe(url, seg):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    sport_df = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                sport_df.append({'Genre' : 'Sports','Segment': seg, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(sport_df)

    return df

# List of URLs and corresponding cities
url_sport_pairs = [
   
   # ("https://www.thehindu.com/sport/", "" 
    ("https://www.thehindu.com/sport/cricket/","cricket"),
    ("https://www.thehindu.com/sport/football/","Football"),
    ("https://www.thehindu.com/sport/hockey/","Hockey"),
    ("https://www.thehindu.com/sport/tennis/","Tennis"),
    ("https://www.thehindu.com/sport/athletics/","Athletics"),
    ("https://www.thehindu.com/sport/motorsport/","Motorsport"),
    ("https://www.thehindu.com/sport/races/","Races"),
    ("https://www.thehindu.com/sport/other-sports/","Other-sports")
]

# List to store all DataFrames
sport_df = []

for url, seg in url_sport_pairs:
    df = scrape_data_and_create_dataframe(url, seg)
    sport_df.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(sport_df, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('Sports_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)


# In[6]:


#OPINION

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def scrape_data_and_create_dataframe(url, seg):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    opinion_df = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                opinion_df.append({'Genre' : 'Opinion','Segment': seg, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(opinion_df)

    return df

# List of URLs and corresponding cities
url_opinion_pairs = [
     ("https://www.thehindu.com/opinion/editorial/", "Editorial"),
    ("https://www.thehindu.com/opinion/columns/","Columns"),
    ("https://www.thehindu.com/opinion/op-ed/","Op-ed"),
    ("https://www.thehindu.com/opinion/cartoon/","Cartoon"),
    ("https://www.thehindu.com/opinion/letters/","Letters"),
    ("https://www.thehindu.com/opinion/interview/","Interview"),
    ("https://www.thehindu.com/opinion/lead/","Lead")

]

# List to store all DataFrames
opinion_df = []

for url, seg in url_opinion_pairs:
    df = scrape_data_and_create_dataframe(url, seg)
    opinion_df.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(opinion_df, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('opinion_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)


# In[7]:


# SCIENCE
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def scrape_data_and_create_dataframe(url, seg):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    sci_df = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                sci_df.append({'Genre' : 'Sci-tech','Segment': seg, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(sci_df)

    return df

# List of URLs and corresponding cities
url_sci_pairs = [
   ("https://www.thehindu.com/sci-tech/science/","Science"),
   ("https://www.thehindu.com/sci-tech/technology/","Technology"),
    ("https://www.thehindu.com/sci-tech/health/","Health"),
    ("https://www.thehindu.com/sci-tech/agriculture/","Agriculture"),
    ("https://www.thehindu.com/sci-tech/energy-and-environment/","Energy-and-environment"),
    ("https://www.thehindu.com/sci-tech/technology/gadgets/","Gadgets"),
    ("https://www.thehindu.com/sci-tech/technology/internet/","Internet")
]

# List to store all DataFrames
sci_df = []

for url, seg in url_sci_pairs:
    df = scrape_data_and_create_dataframe(url, seg)
    sci_df.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(sci_df, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('sci_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)


# In[8]:


# NATIONAL NEWS

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def scrape_data_and_create_dataframe(url, city):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    nation_list = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                nation_list.append({'Genre' : 'National','Segment': city, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(nation_list)

    return df

# List of URLs and corresponding cities
url_national_pairs = [
    ("https://www.thehindu.com/news/national/andhra-pradesh/" , "Andra Pradesh"),
    ("https://www.thehindu.com/news/national/karnataka/" , "Karnataka"),
    ("https://www.thehindu.com/news/national/kerala/" , "Kerala"),
    ("https://www.thehindu.com/news/national/tamil-nadu/" , "Tamil Nadu"),
    ("https://www.thehindu.com/news/national/telangana/" , "Telangana"),
    ("https://www.thehindu.com/news/national/other-states/" , "Other States")
]

# List to store all DataFrames
national_df = []

for url, city in url_national_pairs:
    df = scrape_data_and_create_dataframe(url, city)
    national_df.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(national_df, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('national_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)



# In[9]:


# CITY NEWS

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def scrape_data_and_create_dataframe(url, city):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    data_list = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                data_list.append({'Genre':'City' ,'Segment': city, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    return df

# List of URLs and corresponding cities
url_city_pairs = [
    ("https://www.thehindu.com/news/cities/bangalore/", "Bangalore"),
    ("https://www.thehindu.com/news/cities/chennai/","Chennai"),
    ("https://www.thehindu.com/news/cities/Coimbatore/","Coimbatore"),
    ("https://www.thehindu.com/news/cities/Delhi/", "Delhi"),
    ("https://www.thehindu.com/news/cities/Hyderabad/", "Hyderabad"),
    ("https://www.thehindu.com/news/cities/Kochi/","Kochi"),
    ("https://www.thehindu.com/news/cities/kolkata/","Kolkata"),
    ("https://www.thehindu.com/news/cities/kozhikode/", "Kozhikode"),
    ("https://www.thehindu.com/news/cities/Madurai/","Madurai"),
    ("https://www.thehindu.com/news/cities/Mangalore/","Mangalore"),
    ("https://www.thehindu.com/news/cities/mumbai/","Mumbai"),
    ("https://www.thehindu.com/news/cities/puducherry/","Puducherry"),
    ("https://www.thehindu.com/news/cities/Thiruvananthapuram/","Thiruvananthapuram"),
    ("https://www.thehindu.com/news/cities/Tiruchirapalli/","Tiruchirapalli"),
    ("https://www.thehindu.com/news/cities/Vijayawada/","Vijayawada"),
    ("https://www.thehindu.com/news/cities/Visakhapatnam/","Visakhapatnam")
]

# List to store all DataFrames
all_dfs = []

for url, city in url_city_pairs:
    df = scrape_data_and_create_dataframe(url, city)
    all_dfs.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('city_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)


# In[11]:


# CITY NEWS

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def scrape_data_and_create_dataframe(url, city):
    # Set up the WebDriver (make sure you have the correct webdriver for your browser)
    driver = webdriver.Chrome()

    # List to store data
    entertain_df = []

    # Load the website
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load, you may adjust this based on your needs

    # Get the page source after it has been fully loaded
    page_source = driver.page_source

    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # Now you can use BeautifulSoup methods to extract the data you need
    # For example, let's extract the headlines with both classes
    headlines_with_big = soup.find_all('h3', class_='title big')
    headlines_without_big = soup.find_all('h3', class_='title')

    # Extract data and append to the list
    for h3_element in headlines_with_big + headlines_without_big:
        a_element = h3_element.find('a')
        if a_element:
            strong_element = a_element.find('strong')
            if strong_element:
                title = strong_element.get_text(strip=True)
                processed_title = preprocess_text(title)
                
                # Append data to the list as a dictionary
                entertain_df.append({'Genre':'Entertainment' ,'Segment': entertain, 'Title': processed_title})

    # Close the WebDriver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(entertain_df)

    return df

# List of URLs and corresponding cities
url_entertain_pairs = [
    ("https://www.thehindu.com/entertainment/art/","Art"),
    ("https://www.thehindu.com/entertainment/movies/","Movies")
]

# List to store all DataFrames
all_dfs = []

for url, entertain in url_entertain_pairs:
    df = scrape_data_and_create_dataframe(url, entertain)
    all_dfs.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save combined DataFrame to a CSV file
combined_df.to_csv('Entertainment_news.csv', index=False)

# Display the combined DataFrame
print(combined_df)


# In[ ]:


https://www.thehindu.com/entertainment/art/
https://www.thehindu.com/entertainment/movies/


# In[12]:


# List of CSV files to concatenate
csv_files = [
    'entertainment_news.csv',
    'city_news.csv',
    'national_news.csv',
    'sci_news.csv',
    'opinion_news.csv',
    'sports_news.csv',
    'business_news.csv'
]

# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('collected_news.csv', index=False)

# Read the combined CSV file
read_combined_df = pd.read_csv('collected_news.csv')

# Display the combined DataFrame
print(read_combined_df)


# In[13]:


read_combined_df.isnull().sum()


# In[ ]:




