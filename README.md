# 📰 Project Report: Bing News Scraper and Sentiment Analysis

## 📚 Introduction
The project aims to extract news articles related to intellectual property and patent law using Bing News. The data is processed to remove duplicates, cleaned, and analyzed for trends, sentiment, and word frequency. The goal is to understand how intellectual property is represented in the news and identify patterns, including the sentiment of articles and their relevance to specific keywords. Additionally, the project includes summarizing article content and creating a predictive sentiment model.

---

## 🛠️ Methodology

### 🌐 Web Scraping
- **🔑 Keywords**: The scraper focuses on keywords such as "intellectual property," "patent lawyer," "IP enforcement," and others.
- **🛠️ Tool Used**: Selenium was used with ChromeDriver to automate data collection.
- **📥 Data Collection**: News titles and URLs were scraped for 10 pages of search results per keyword.
- **📂 Output**: A CSV file containing the scraped data.

### 🧹 Data Cleaning
- **❌ Duplicate Removal**: Articles with duplicate URLs were removed to ensure data quality.
- **🔍 Keyword Filtering**: Titles containing relevant keywords were retained.
- **📊 Dataset Overview**: The cleaned dataset had columns for titles, links, and associated keywords.

### 📈 Data Analysis
- **🧠 Sentiment Analysis**: Using TextBlob, the polarity of article titles was calculated to classify them as positive, neutral, or negative.
- **📌 Word Frequency**: The most common words in titles were identified.
- **📊 Visualization**:
  - 📂 Distribution of articles by keyword.
  - 😐 Sentiment distribution of titles.
  - 📉 Scatter plot of sentiment vs. word count.

### ✂️ Content Summarization
- **🤖 Tool**: The DistilBART summarization model was used to summarize the content of each article.
- **💻 Platform**: Summarization was conducted on Kaggle’s GPU environment to leverage fast computation.
- **📄 Output**: Summarized content for each article.

### 📊 Predictive Model
- **🤖 Model**: Random Forest Classifier was trained to predict sentiment based on the TF-IDF representation of article titles.
- **✅ Accuracy**: The model achieved satisfactory accuracy on the test set.

### 📄 Content Extraction
- 🖋️ Full article content was extracted using BeautifulSoup and summarized using NLP techniques.

---

## 🎯 Results

### 📂 Article Distribution
- Articles were unevenly distributed among keywords, with "intellectual property" and "patent lawyer" having the highest counts.

### 😐 Sentiment Analysis
- Most articles were neutral, followed by positive 😊, and very few were negative 😞.
- Positive and negative word clouds highlighted distinct themes.

### 📝 Keyword Insights
- Average sentiment scores varied across keywords, reflecting differences in the tone of news coverage.

### 📈 Model Performance
- The sentiment classification model achieved reliable accuracy and demonstrated the potential for predictive analysis in news data.

### ✂️ Summarization
- Summarized articles provided concise insights into the content, aiding quick understanding.

### 📊 Key Visualizations
- **📊 Bar Chart**: Distribution of articles by keyword.
- **📊 Histogram**: Sentiment distribution.
- **📉 Scatter Plot**: Sentiment vs. word count.
- **☁️ Word Clouds**: Positive and negative word themes.

---

## 🏁 Conclusion
This project successfully demonstrates the application of web scraping, NLP, and machine learning for analyzing news data. The insights derived can help understand trends in intellectual property and patent law coverage. The sentiment analysis and summarization further enhance the usability of the data for stakeholders such as legal professionals and policymakers.

---

## 🚀 Future Work
- 🔄 Automate daily scraping for up-to-date analysis.
- 🧠 Enhance the sentiment analysis model by including advanced techniques such as BERT.
- 📊 Integrate the project into a dashboard for real-time monitoring and insights.

---

## 📚 References
- 📜 Selenium Documentation  
- 😊 TextBlob Sentiment Analysis  
- 🤖 Hugging Face Transformers Library  
- 📊 Matplotlib and WordCloud Libraries
