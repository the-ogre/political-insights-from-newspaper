import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import json
from datetime import datetime

def perform_sentiment_analysis(df, model_name="distilbert-base-uncased-finetuned-sst-2-english", text_column='News Content'):
    """
    Analyze sentiment in news articles using DistilBERT.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news articles
        model_name (str): Name of the Hugging Face model to use
        text_column (str): Name of the column containing the text to analyze
        
    Returns:
        pd.DataFrame: DataFrame with additional sentiment analysis columns
    """
    print(f"Initializing sentiment analysis model: {model_name}")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Initialize the sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model=model, 
                                tokenizer=tokenizer,
                                device=0 if torch.cuda.is_available() else -1)
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add sentiment columns
    result_df['Sentiment_Label'] = None
    result_df['Sentiment_Score'] = None
    result_df['Sentiment_JSON'] = None
    
    print("Starting sentiment analysis...")
    start_time = time.time()
    
    # Process each article
    for idx, row in result_df.iterrows():
        text = row[text_column]
        
        # Skip empty texts
        if not isinstance(text, str) or len(text) == 0:
            continue
        
        try:
            # Many models have a token limit, so process in chunks if needed
            if len(text) > 512:  # Arbitrary cutoff
                text = text[:512]  # Just use beginning of text for sentiment
            
            sentiment = sentiment_analyzer(text)[0]
            
            # Store the sentiment analysis results
            result_df.at[idx, 'Sentiment_Label'] = sentiment['label']
            result_df.at[idx, 'Sentiment_Score'] = sentiment['score']
            result_df.at[idx, 'Sentiment_JSON'] = json.dumps(sentiment)
            
            print(f"Analyzed sentiment for article {idx+1}/{len(result_df)}")
        except Exception as e:
            print(f"Error analyzing sentiment for article {idx}: {e}")
    
    end_time = time.time()
    print(f"Sentiment analysis completed in {end_time - start_time:.2f} seconds")
    
    return result_df

def visualize_sentiment_results(df):
    """
    Create visualizations for the sentiment analysis results.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment information
    
    Returns:
        None: Saves visualization files to disk
    """
    print("Generating sentiment analysis visualizations...")
    
    # Filter out rows with missing sentiment data
    filtered_df = df.dropna(subset=['Sentiment_Label', 'Sentiment_Score'])
    
    if len(filtered_df) == 0:
        print("No valid sentiment data to visualize")
        return
    
    # Set up the figure
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Sentiment distribution pie chart
    plt.subplot(2, 2, 1)
    sentiment_counts = filtered_df['Sentiment_Label'].value_counts()
    
    colors = ['#5cb85c', '#d9534f'] if 'POSITIVE' in sentiment_counts.index else ['#d9534f', '#5cb85c']
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors,
           startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Sentiment Distribution', fontsize=14)
    
    # Plot 2: Sentiment scores bar chart
    plt.subplot(2, 2, 2)
    
    # Sort by sentiment score for better visualization
    sorted_df = filtered_df.sort_values('Sentiment_Score', ascending=False)
    
    # Create a color map based on sentiment
    colors = ['#5cb85c' if label == 'POSITIVE' else '#d9534f' for label in sorted_df['Sentiment_Label']]
    
    plt.bar(range(len(sorted_df)), sorted_df['Sentiment_Score'], color=colors)
    plt.xlabel('Article Index', fontsize=12)
    plt.ylabel('Confidence Score', fontsize=12)
    plt.title('Sentiment Confidence Scores', fontsize=14)
    plt.ylim(0, 1)
    
    # Plot 3: Headline Sentiment
    plt.subplot(2, 2, 3)
    
    # Create a scatter plot of sentiment scores
    # Use a diverging color map
    plt.scatter(range(len(filtered_df)), 
              filtered_df['Sentiment_Score'], 
              c=filtered_df['Sentiment_Score'], 
              cmap='RdYlGn', 
              s=100, 
              alpha=0.7)
    
    plt.colorbar(label='Sentiment Score')
    plt.xlabel('Article Index', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.title('Article Sentiment Score Distribution', fontsize=14)
    plt.ylim(0, 1)
    
    # Plot 4: Example headlines with sentiment
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    plt.text(0, 1.0, "Example Headlines with Sentiment", fontsize=14, fontweight='bold')
    
    # Get a few positive and negative examples
    positive_examples = filtered_df[filtered_df['Sentiment_Label'] == 'POSITIVE'].head(3)
    negative_examples = filtered_df[filtered_df['Sentiment_Label'] == 'NEGATIVE'].head(3)
    
    y_pos = 0.9
    plt.text(0, y_pos, "Positive Headlines:", fontsize=12, fontweight='bold')
    y_pos -= 0.05
    
    for _, row in positive_examples.iterrows():
        headline = row['Headline']
        score = row['Sentiment_Score']
        
        # Truncate long headlines
        if len(headline) > 60:
            headline = headline[:57] + '...'
            
        plt.text(0, y_pos, f"• {headline}", fontsize=10)
        plt.text(0.8, y_pos, f"{score:.2f}", fontsize=10, color='green')
        y_pos -= 0.05
    
    y_pos -= 0.05
    plt.text(0, y_pos, "Negative Headlines:", fontsize=12, fontweight='bold')
    y_pos -= 0.05
    
    for _, row in negative_examples.iterrows():
        headline = row['Headline']
        score = row['Sentiment_Score']
        
        # Truncate long headlines
        if len(headline) > 60:
            headline = headline[:57] + '...'
            
        plt.text(0, y_pos, f"• {headline}", fontsize=10)
        plt.text(0.8, y_pos, f"{score:.2f}", fontsize=10, color='red')
        y_pos -= 0.05
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=300)
    plt.close()
    
    print("Sentiment analysis visualization saved as 'sentiment_analysis.png'")

def analyze_sentiment_by_time(df):
    """
    Analyze sentiment trends over time.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment information and timestamps
        
    Returns:
        pd.DataFrame: DataFrame with aggregated sentiment by date
    """
    # Ensure we have timestamps and sentiment data
    if 'Timestamp' not in df.columns or 'Sentiment_Label' not in df.columns:
        print("Missing required columns for time-based sentiment analysis")
        return None
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_dtype(df['Timestamp']):
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            return None
    
    # Extract date from timestamp
    df['Date'] = df['Timestamp'].dt.date
    
    # Group by date and calculate sentiment metrics
    date_sentiment = df.groupby('Date').agg({
        'Sentiment_Label': lambda x: (x == 'POSITIVE').mean(),  # Proportion of positive sentiment
        'Sentiment_Score': 'mean',  # Average sentiment score
        'Headline': 'count'  # Number of articles
    }).reset_index()
    
    # Rename columns for clarity
    date_sentiment.rename(columns={
        'Sentiment_Label': 'Positive_Ratio',
        'Headline': 'Article_Count'
    }, inplace=True)
    
    return date_sentiment

def visualize_sentiment_over_time(date_sentiment):
    """
    Visualize sentiment trends over time.
    
    Args:
        date_sentiment (pd.DataFrame): DataFrame with aggregated sentiment by date
        
    Returns:
        None: Saves visualization file to disk
    """
    if date_sentiment is None or len(date_sentiment) == 0:
        print("No valid data for time-based sentiment visualization")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Positive sentiment ratio over time
    ax1.plot(date_sentiment['Date'], date_sentiment['Positive_Ratio'], 
            marker='o', linestyle='-', color='green', linewidth=2)
    
    ax1.set_ylabel('Positive Sentiment Ratio', fontsize=12)
    ax1.set_title('Positive Sentiment Trend Over Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add a horizontal line at 0.5 for reference
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 2: Article count by date
    ax2.bar(date_sentiment['Date'], date_sentiment['Article_Count'], color='blue', alpha=0.6)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Article Count', fontsize=12)
    ax2.set_title('Number of Articles by Date', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png', dpi=300)
    plt.close()
    
    print("Time-based sentiment visualization saved as 'sentiment_over_time.png'")

# Example usage
if __name__ == "__main__":
    # Load the preprocessed data
    try:
        # Try to load the summarized articles with summaries first
        input_csv = "summarized_articles.csv"
        df = pd.read_csv(input_csv)
        text_column = 'Summary'  # Use the summaries for sentiment analysis
    except:
        # Fall back to the original processed data
        input_csv = "indian_express_processed.csv"
        df = pd.read_csv(input_csv)
        text_column = 'News Content'
    
    print(f"Loaded data from {input_csv}, using {text_column} for sentiment analysis")
    
    # Perform sentiment analysis
    sentiment_df = perform_sentiment_analysis(df, text_column=text_column)
    
    # Visualize the sentiment results
    visualize_sentiment_results(sentiment_df)
    
    # Analyze sentiment over time
    date_sentiment = analyze_sentiment_by_time(sentiment_df)
    if date_sentiment is not None:
        visualize_sentiment_over_time(date_sentiment)
    
    # Save the results
    sentiment_df.to_csv("sentiment_analyzed_articles.csv", index=False)
    print("Sentiment analyzed articles saved to 'sentiment_analyzed_articles.csv'")
    
    # Display sample sentiment results
    if len(sentiment_df) > 0:
        print("\nSample Sentiment Analysis Results:")
        for i in range(min(3, len(sentiment_df))):
            sample = sentiment_df.iloc[i]
            print(f"\nArticle {i+1}:")
            print(f"Headline: {sample['Headline']}")
            print(f"Sentiment: {sample['Sentiment_Label']} (Score: {sample['Sentiment_Score']:.4f})")