# %%
import this

# %%


# %%
keyword = ["intellectual property", "patent lawyer", "IP enforcement", "inventor", 
            "patent application", "technology patents", "trademark law", "IP litigation"]


# %% [markdown]
# Using Bing to scrap data

# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Set up Selenium with webdriver-manager
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    service = Service(ChromeDriverManager().install())  # Automatically download and set up ChromeDriver
    return webdriver.Chrome(service=service, options=chrome_options)

# Function to scrape Bing News
def scrape_bing_news(keywords, num_pages=10):
    driver = get_driver()
    articles = []

    for keyword in keywords:
        print(f"Scraping articles for keyword: {keyword}")
        for page in range(1, num_pages + 1):
            url = f"https://www.bing.com/news/search?q={keyword.replace(' ', '+')}&first={page * 10}"
            driver.get(url)
            time.sleep(3)  # Wait for the page to load

            # Scrape article titles and links
            news_cards = driver.find_elements(By.CSS_SELECTOR, 'a.title')
            for card in news_cards:
                try:
                    title = card.text
                    link = card.get_attribute('href')
                    articles.append({"Keyword": keyword, "Title": title, "Link": link})
                except Exception as e:
                    print(f"Error scraping a news card: {e}")

    driver.quit()
    return pd.DataFrame(articles)

# Keywords to search
keywords = ["intellectual property", "patent lawyer", "IP enforcement", "inventor", 
            "patent application", "technology patents", "trademark law", "IP litigation"]

# Run the scraper
df = scrape_bing_news(keywords)
print(df)

# Save results to a CSV file
df.to_csv("bing_news_articles.csv", index=False)


# %% [markdown]
# Data Cleaning

# %%
df = pd.read_csv('/Users/abhinavyadav/VS Code/MachineLearning/Projects/Complete ML Project/bing_news_articles.csv')
df

# %%
df.shape

# %%
#drop duplicates values
df = df.drop_duplicates(subset=['Link'])

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %%
df.isna().sum()

# %%
df.shape

# %%
df

# %%
df.reset_index(inplace=True)
df.rename(columns={"index": "Index"}, inplace=True)

# %%
df

# %%


# %%
# Adjust pandas display options to show the full content
pd.set_option('display.max_colwidth', None)
df[df['Index'] == 8]

# %%
# Keep only articles that contain relevant keywords in the title
relevant_keywords = ["intellectual property", "patent lawyer", "IP enforcement", "inventor", "patent application", "technology patents", "trademark law", "IP litigation"]
df_filtered = df[df['Title'].str.contains('|'.join(relevant_keywords), case=False, na=False)]


# %%
df_filtered  

# %%
df_filtered.reset_index(drop=True, inplace=True)

# %%
df_filtered.head()

# %%
df_filtered.to_csv('new_df.csv', index=False)

# %%
# Count how many articles belong to each keyword
keyword_counts = df_filtered.groupby('Keyword').size()
print(keyword_counts)


# %% [markdown]
# Visulaize

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
keyword_counts.plot(kind='bar', color='skyblue')
plt.title("Distribution of Articles by Keyword")
plt.xlabel("Keyword")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
from collections import Counter
import re

# Tokenize and count words in titles
all_titles = " ".join(df_filtered["Title"]).lower()
words = re.findall(r'\w+', all_titles)
word_counts = Counter(words)

# Display top 10 most common words
top_words = word_counts.most_common(10)
print(top_words)


# %%
!pip install textblob

# %%
from textblob import TextBlob

# Add sentiment column based on title
df_filtered['Sentiment'] = df_filtered['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['Sentiment'], bins=20, color='lightgreen')
plt.title("Sentiment Distribution of Article Titles")
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()


# %% [markdown]
# '''
# This show's that most number of article is neutral, 
# Then some are positive and
# Very fews are negative
# '''

# %% [markdown]
# ## Sentiment vs. Word Count

# %%
# Add word count column
df_filtered['Word_Count'] = df_filtered['Title'].apply(lambda x: len(x.split()))

# Scatter plot: Word count vs Sentiment
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Word_Count'], df_filtered['Sentiment'], alpha=0.6, color='purple')
plt.title("Sentiment vs. Word Count")
plt.xlabel("Word Count")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Average Sentiment by Keyword
# 

# %%
# Average sentiment by keyword
average_sentiment = df_filtered.groupby('Keyword')['Sentiment'].mean().sort_values()

# Plot average sentiment by keyword
plt.figure(figsize=(12, 6))
average_sentiment.plot(kind='bar', color='teal')
plt.title("Average Sentiment by Keyword")
plt.xlabel("Keyword")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Word Cloud

# %%
from wordcloud import WordCloud

# Positive titles word cloud
positive_titles = ' '.join(df_filtered[df_filtered['Sentiment'] > 0]['Title'])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_titles)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Positive Titles")
plt.show()

# Negative titles word cloud
negative_titles = ' '.join(df_filtered[df_filtered['Sentiment'] < 0]['Title'])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_titles)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Negative Titles")
plt.show()


# %% [markdown]
# ## Model Building

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert titles into TF-IDF features
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(df_filtered['Title']).toarray()


# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example: Predicting sentiment (binary: positive or negative)
df_filtered['Sentiment_Label'] = df_filtered['Sentiment'].apply(lambda x: 1 if x > 0 else 0)  # 1 for positive, 0 for negative

X_train, X_test, y_train, y_test = train_test_split(X, df_filtered['Sentiment_Label'], test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# %%


# %% [markdown]
# ## Scraping Content using the links

# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm  # For progress tracking

# Function to extract article content from a URL
def extract_article_content(url, retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text from <p> tags
                paragraphs = soup.find_all('p')
                article = " ".join([p.get_text() for p in paragraphs])
                return article.strip() if article else "No content found"
            else:
                return f"Failed to fetch. Status code: {response.status_code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error: {e}"
    return "Failed after multiple retries"

# Load the CSV file
input_csv_file = "/Users/abhinavyadav/VS Code/MachineLearning/Projects/Complete ML Project/new_df.csv"  
df = pd.read_csv(input_csv_file)

# Ensure the DataFrame has a "Link" column
if 'Link' not in df.columns:
    raise ValueError("The CSV file must have a 'Link' column containing the URLs.")

# Add a 'Content' column using the extract_article_content function
print("Starting to scrape articles...")
tqdm.pandas()  # Enable progress bar for pandas operations
df['Content'] = df['Link'].progress_apply(extract_article_content)

# Save the updated DataFrame to a new CSV file
output_csv_file = "articles_with_content.csv"
df.to_csv(output_csv_file, index=False)

print(f"Scraping completed. Results saved to {output_csv_file}.")


# %%
temp_df = pd.read_csv('/Users/abhinavyadav/VS Code/MachineLearning/Projects/Complete ML Project/articles_with_content.csv')
temp_df.head()

# %% [markdown]
# ## Summarizing the text in content

# %%
!pip install transformers

# %%
!pip install torch torchvision torchaudio


# %%
from transformers import pipeline

# Create a summarization pipeline using PyTorch
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

# Example text for summarization
article_text = """
The intellectual property (IP) landscape has been rapidly evolving in recent years, especially in technology sectors.
The advent of AI, blockchain, and other innovative technologies has introduced new challenges and opportunities for IP lawyers.
In this article, we explore the implications of these developments and how IP law is adapting to keep up with technological advancements.
"""

# Summarize the text
summary = summarizer(article_text, max_length=50, min_length=50, do_sample=False)

# Print the summary
print(summary[0]['summary_text'])


# %%
df.head()

# %% [markdown]
# ## I use this model on kaggle notebook to summarize the content text using GPU T4x2

# %%
import pandas as pd
from transformers import pipeline

# Create a summarization pipeline using PyTorch
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

# Read the CSV file containing the 'content' column
df = pd.read_csv('/Users/abhinavyadav/VS Code/MachineLearning/Projects/Complete ML Project/articles_with_content.csv')  

# Function to summarize content
def summarize_content(content):
    # Ensure content is not empty or NaN
    if pd.notna(content):
        summary = summarizer(content, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return None

# Apply the summarization function to each row's content
df['summary'] = df['Content'].apply(summarize_content)

# Save the results to a new CSV file (optional)
df.to_csv('summarized_output.csv', index=False)

# Print the first few rows of the output
print(df[['content', 'summary']].head())


# %%
summarized_df = pd.read_csv('/Users/abhinavyadav/VS Code/MachineLearning/Projects/Complete ML Project/summarized_output.csv')
summarized_df.drop(columns='Content', inplace=True)

# %%
summarized_df.head()

# %%



