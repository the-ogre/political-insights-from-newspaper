import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast, AutoModelForTokenClassification
import torch
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Set up plot styling
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV data
def load_data(file_path):
    """Load and preprocess the news articles dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} articles")
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%B %d, %Y %H:%M IST")
    
    # Add a date column for easier analysis
    df['Date'] = df['Timestamp'].dt.date
    
    # Add a week column for weekly analysis
    df['Week'] = df['Timestamp'].dt.isocalendar().week
    
    return df

# Text summarization with BERT
def summarize_texts(texts, max_length=150, min_length=40):
    """Generate summaries for a list of texts using a BERT-based model."""
    print("Performing text summarization...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    summaries = []
    for text in texts:
        # BART models usually have a limit of 1024 tokens, so truncate if needed
        if len(text) > 1024:
            text = text[:1024]
        
        # Skip empty or very short texts
        if len(text) < 50:
            summaries.append("")
            continue
            
        summary = summarizer(text, max_length=max_length, min_length=min_length, 
                            do_sample=False)[0]['summary_text']
        summaries.append(summary)
        
    return summaries

# Sentiment analysis with DistilBERT
def analyze_sentiment(texts):
    """Perform sentiment analysis on a list of texts using DistilBERT."""
    print("Performing sentiment analysis...")
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english")
    
    results = []
    for text in texts:
        # Many models have a token limit, so process in chunks if needed
        if len(text) > 500:  # Arbitrary cutoff
            text = text[:500]  # Just use beginning of text for sentiment
            
        sentiment = sentiment_analyzer(text)[0]
        results.append({
            'label': sentiment['label'],
            'score': sentiment['score']
        })
        
    return results

# Topic modeling with BERTopic
def model_topics(texts, n_topics=5):
    """Extract topics from texts using BERTopic."""
    print("Performing topic modeling...")
    # Initialize BERTopic model
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.9)
    topic_model = BERTopic(vectorizer_model=vectorizer, 
                          calculate_probabilities=True, 
                          nr_topics=n_topics)
    
    # Fit the model
    topics, probabilities = topic_model.fit_transform(texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Get top topics
    top_topics = []
    for topic in topic_info.loc[topic_info['Topic'] != -1]['Topic']:
        words = topic_model.get_topic(topic)
        top_words = [word for word, _ in words[:5]]
        top_topics.append({
            'id': topic,
            'words': top_words
        })
    
    return topics, top_topics, topic_model

# Weekly sentiment analysis
def analyze_weekly_sentiment(df):
    """Analyze sentiment trends by week."""
    print("Analyzing weekly sentiment trends...")
    weekly_sentiment = df.groupby('Week').apply(
        lambda x: {
            'positive': sum(1 for s in x['Sentiment'] if s['label'] == 'POSITIVE'),
            'negative': sum(1 for s in x['Sentiment'] if s['label'] == 'NEGATIVE'),
            'positive_ratio': sum(1 for s in x['Sentiment'] if s['label'] == 'POSITIVE') / len(x),
            'count': len(x)
        }
    ).reset_index()
    
    return weekly_sentiment

# Named Entity Recognition
def perform_ner(texts, target_entities=None):
    """Extract named entities from texts, focusing on specified target entities."""
    print("Performing Named Entity Recognition...")
    if target_entities is None:
        target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    # Initialize NER pipeline
    ner = pipeline("ner", model="dslim/bert-base-NER")
    
    entity_counts = {entity: 0 for entity in target_entities}
    
    for text in texts:
        # Process text in chunks due to token limits
        chunk_size = 400
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in chunks:
            entities = ner(chunk)
            
            # Group consecutive entities with the same type
            grouped_entities = []
            current_entity = None
            
            for entity in entities:
                if current_entity is None:
                    current_entity = entity
                elif (entity['entity'] == current_entity['entity'] and 
                      entity['start'] == current_entity['end'] + 1):
                    current_entity['word'] += ' ' + entity['word']
                    current_entity['end'] = entity['end']
                else:
                    grouped_entities.append(current_entity)
                    current_entity = entity
            
            if current_entity:
                grouped_entities.append(current_entity)
            
            # Count occurrences of target entities
            for entity in grouped_entities:
                entity_text = entity['word'].lower()
                
                # Check for target entities
                for target in target_entities:
                    if target.lower() in entity_text:
                        entity_counts[target] += 1
                        break
    
    return entity_counts

# Visualize text summarization results
def visualize_summarization(df):
    """Create visualization comparing original text length to summary length."""
    print("Visualizing text summarization results...")
    
    # Calculate text lengths
    df['Content_Length'] = df['News Content'].apply(len)
    df['Summary_Length'] = df['Summary'].apply(len)
    df['Compression_Ratio'] = df['Summary_Length'] / df['Content_Length']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Length comparison
    indices = range(len(df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in indices], df['Content_Length'], width, label='Original Content')
    ax1.bar([i + width/2 for i in indices], df['Summary_Length'], width, label='Summary')
    
    ax1.set_xlabel('Article Index')
    ax1.set_ylabel('Character Count')
    ax1.set_title('Original vs Summary Length Comparison')
    ax1.set_xticks(indices)
    ax1.legend()
    
    # Plot 2: Compression ratio
    ax2.bar(indices, df['Compression_Ratio'] *.100)
    ax2.set_xlabel('Article Index')
    ax2.set_ylabel('Compression Ratio (%)')
    ax2.set_title('Summary Compression Ratio')
    ax2.set_xticks(indices)
    
    plt.tight_layout()
    plt.savefig('summarization_analysis.png')
    plt.close()
    
    return 'summarization_analysis.png'

# Visualize sentiment analysis results
def visualize_sentiment(df):
    """Create visualizations for sentiment analysis results."""
    print("Visualizing sentiment analysis results...")
    
    # Extract sentiment scores
    df['Sentiment_Score'] = df['Sentiment'].apply(lambda x: x['score'])
    df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: x['label'])
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Sentiment distribution
    sentiment_counts = df['Sentiment_Label'].value_counts()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           colors=['green', 'red'] if 'POSITIVE' in sentiment_counts.index else ['red'])
    ax1.set_title('Sentiment Distribution')
    
    # Plot 2: Sentiment scores
    sns.barplot(x=df.index, y='Sentiment_Score', hue='Sentiment_Label', data=df, ax=ax2)
    ax2.set_xlabel('Article Index')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Sentiment Analysis Confidence')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png')
    plt.close()
    
    return 'sentiment_analysis.png'

# Visualize topic modeling results
def visualize_topics(df, topic_model):
    """Create visualizations for topic modeling results."""
    print("Visualizing topic modeling results...")
    
    # Get topic distribution
    topic_counts = df['Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    # Get topic labels
    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    
    for _, row in topic_info.iterrows():
        if row['Topic'] != -1:  # Skip outlier topic
            words = [word for word, _ in topic_model.get_topic(row['Topic'])[:3]]
            topic_labels[row['Topic']] = f"Topic {row['Topic']}: {', '.join(words)}"
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Topic distribution
    sns.barplot(x='Topic', y='Count', data=topic_counts[topic_counts['Topic'] != -1], ax=ax1)
    ax1.set_xlabel('Topic ID')
    ax1.set_ylabel('Article Count')
    ax1.set_title('Topic Distribution')
    
    # Plot 2: Topic visualization
    topic_model.visualize_topics(top_n_topics=5).write_html("topic_visualization.html")
    
    # Generate a wordcloud for the top topic
    if len(topic_counts) > 0 and topic_counts['Topic'].iloc[0] != -1:
        top_topic = topic_counts['Topic'].iloc[0]
        topic_words = dict(topic_model.get_topic(top_topic))
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
        
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title(f'Word Cloud for {topic_labels.get(top_topic, f"Topic {top_topic}")}')
    
    plt.tight_layout()
    plt.savefig('topic_modeling.png')
    plt.close()
    
    return 'topic_modeling.png'

# Visualize weekly sentiment analysis
def visualize_weekly_sentiment(weekly_sentiment):
    """Create visualization for weekly sentiment trends."""
    print("Visualizing weekly sentiment trends...")
    
    # Prepare data for plotting
    weeks = weekly_sentiment['Week']
    positive_ratios = [data['positive_ratio'] for data in weekly_sentiment[0]]
    counts = [data['count'] for data in weekly_sentiment[0]]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Positive sentiment ratio by week
    ax1.plot(weeks, positive_ratios, marker='o', linestyle='-', color='green')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Positive Sentiment Ratio')
    ax1.set_title('Weekly Positive Sentiment Trend')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # Plot 2: Article count by week
    ax2.bar(weeks, counts, color='blue', alpha=0.7)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Article Count')
    ax2.set_title('Weekly Article Count')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('weekly_sentiment.png')
    plt.close()
    
    return 'weekly_sentiment.png'

# Visualize named entity recognition results
def visualize_ner(entity_counts):
    """Create visualization for named entity counts."""
    print("Visualizing named entity recognition results...")
    
    entities = list(entity_counts.keys())
    counts = list(entity_counts.values())
    
    # Create a color map
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(entities, counts, color=colors)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.xlabel('Entity')
    plt.ylabel('Mention Count')
    plt.title('Key Political Entity Mentions in News Articles')
    plt.tight_layout()
    
    plt.savefig('ner_analysis.png')
    plt.close()
    
    return 'ner_analysis.png'

# Main function to run the entire analysis
def main(file_path):
    """Run the complete analysis pipeline on the news articles dataset."""
    # Load and preprocess data
    df = load_data(file_path)
    
    # 1. Text Summarization with BERT
    df['Summary'] = summarize_texts(df['News Content'].tolist())
    
    # 2. Sentiment Analysis with DistilBERT
    df['Sentiment'] = analyze_sentiment(df['News Content'].tolist())
    
    # 3. Topic Modeling with BERTopic
    topics, top_topics, topic_model = model_topics(df['News Content'].tolist())
    df['Topic'] = topics
    
    # 4. Weekly Sentiment Analysis
    weekly_sentiment = analyze_weekly_sentiment(df)
    
    # 5. Named Entity Recognition for Key Figures
    entity_counts = perform_ner(df['News Content'].tolist())
    
    # Generate visualizations
    summary_viz = visualize_summarization(df)
    sentiment_viz = visualize_sentiment(df)
    topic_viz = visualize_topics(df, topic_model)
    weekly_viz = visualize_weekly_sentiment(weekly_sentiment)
    ner_viz = visualize_ner(entity_counts)
    
    print("Analysis complete. Visualization files generated.")
    
    # Return the results
    return {
        'dataframe': df,
        'weekly_sentiment': weekly_sentiment,
        'entity_counts': entity_counts,
        'visualizations': {
            'summary': summary_viz,
            'sentiment': sentiment_viz,
            'topics': topic_viz,
            'weekly': weekly_viz,
            'ner': ner_viz
        }
    }

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "indian_express_articles.csv"
    results = main(file_path)