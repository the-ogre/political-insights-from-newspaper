import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel
import torch
import time
import plotly.express as px
from wordcloud import WordCloud
import itertools
import umap
from hdbscan import HDBSCAN
import plotly.io as pio
import warnings
warnings.filterwarnings("ignore")

def perform_topic_modeling(df, text_column='News Content', n_topics=5):
    """
    Extract topics from news articles using BERTopic.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news articles
        text_column (str): Name of the column containing the text to analyze
        n_topics (int): Number of topics to extract
        
    Returns:
        tuple: (DataFrame with topic assignments, BERTopic model)
    """
    print(f"Starting topic modeling with BERTopic (target: {n_topics} topics)")
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Prepare the texts
    texts = result_df[text_column].tolist()
    
    # Remove any None or empty strings
    texts = [text for text in texts if isinstance(text, str) and text.strip()]
    
    if len(texts) == 0:
        print("No valid texts found for topic modeling")
        return result_df, None
    
    start_time = time.time()
    
    # Configure BERTopic
    # We'll use more standard techniques for small datasets
    # For larger datasets, you might use the default BERT embedding
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.9)
    
    try:
        # Initialize and fit the model - try automatic topic reduction
        topic_model = BERTopic(
            language="english",
            nr_topics=n_topics,
            vectorizer_model=vectorizer,
            calculate_probabilities=True,
            verbose=True
        )
        
        # Fit the model
        topics, probs = topic_model.fit_transform(texts)
        
        end_time = time.time()
        print(f"Topic modeling completed in {end_time - start_time:.2f} seconds")
        
        # Add topics to the DataFrame
        result_df['Topic'] = None
        result_df['Topic_Probability'] = None
        result_df['Topic_Keywords'] = None
        
        # Map topics back to the original DataFrame
        for i, (topic, prob) in enumerate(zip(topics, probs)):
            if i < len(result_df):
                result_df.at[i, 'Topic'] = int(topic)
                
                # Get the probability of the assigned topic
                max_prob_index = np.argmax(prob)
                result_df.at[i, 'Topic_Probability'] = prob[max_prob_index]
                
                # Get the keywords for this topic
                if topic != -1:  # -1 is the outlier topic
                    keywords = [word for word, _ in topic_model.get_topic(topic)]
                    result_df.at[i, 'Topic_Keywords'] = ', '.join(keywords[:5])
                else:
                    result_df.at[i, 'Topic_Keywords'] = 'Outlier/Miscellaneous'
        
        # Get information about the topics
        topic_info = topic_model.get_topic_info()
        print(f"Identified {len(topic_info[topic_info['Topic'] != -1])} topics")
        
        for index, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                top_words = [word for word, _ in topic_model.get_topic(topic_id)]
                print(f"Topic {topic_id}: {', '.join(top_words[:5])}")
        
        return result_df, topic_model
        
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        # Try with a simpler approach if the first one fails
        print("Retrying with basic parameters...")
        
        try:
            # Simplified configuration for small datasets
            topic_model = BERTopic(language="english", nr_topics="auto")
            
            # Fit the model
            topics, probs = topic_model.fit_transform(texts)
            
            end_time = time.time()
            print(f"Topic modeling completed in {end_time - start_time:.2f} seconds")
            
            # Add topics to the DataFrame
            result_df['Topic'] = None
            result_df['Topic_Probability'] = None
            result_df['Topic_Keywords'] = None
            
            # Map topics back to the original DataFrame
            for i, topic in enumerate(topics):
                if i < len(result_df):
                    result_df.at[i, 'Topic'] = int(topic)
                    
                    # Get the keywords for this topic
                    if topic != -1:  # -1 is the outlier topic
                        keywords = [word for word, _ in topic_model.get_topic(topic)]
                        result_df.at[i, 'Topic_Keywords'] = ', '.join(keywords[:5])
                    else:
                        result_df.at[i, 'Topic_Keywords'] = 'Outlier/Miscellaneous'
            
            return result_df, topic_model
            
        except Exception as e:
            print(f"Error in simplified topic modeling: {e}")
            return result_df, None

def visualize_topic_modeling_results(df, topic_model):
    """
    Create visualizations for the topic modeling results.
    
    Args:
        df (pd.DataFrame): DataFrame with topic assignments
        topic_model (BERTopic): Fitted BERTopic model
        
    Returns:
        None: Saves visualization files to disk
    """
    print("Generating topic modeling visualizations...")
    
    if topic_model is None:
        print("No topic model available for visualization")
        return
    
    # Filter out rows with missing topic data
    filtered_df = df.dropna(subset=['Topic'])
    
    if len(filtered_df) == 0:
        print("No valid topic data to visualize")
        return
    
    # Get topic info from the model
    topic_info = topic_model.get_topic_info()
    
    # Extract non-outlier topics
    relevant_topics = topic_info[topic_info['Topic'] != -1]
    
    # 1. Topic Distribution
    plt.figure(figsize=(12, 8))
    
    topic_counts = filtered_df['Topic'].value_counts()
    
    # Filter out the outlier topic for the visualization
    topic_counts = topic_counts[topic_counts.index != -1]
    
    # Create label mappings
    topic_labels = {}
    for topic in topic_counts.index:
        if topic != -1:
            words = topic_model.get_topic(topic)
            top_words = [word for word, _ in words[:3]]
            topic_labels[topic] = f"Topic {topic}: {', '.join(top_words)}"
    
    plt.barh([topic_labels.get(topic, f"Topic {topic}") for topic in topic_counts.index], 
            topic_counts.values, 
            color=plt.cm.tab10(range(len(topic_counts))))
    
    plt.xlabel('Number of Articles', fontsize=12)
    plt.title('Distribution of Articles Across Topics', fontsize=14)
    plt.gca().invert_yaxis()  # To have the most frequent at the top
    
    plt.tight_layout()
    plt.savefig('topic_distribution.png', dpi=300)
    plt.close()
    
    # 2. Topic Word Clouds
    n_topics = min(len(relevant_topics), 4)  # Limit to 4 topics for visibility
    
    if n_topics > 0:
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, topic_id in enumerate(relevant_topics['Topic'][:n_topics]):
            if i < len(axes):
                # Get the words and weights for this topic
                words = topic_model.get_topic(topic_id)
                
                if words:
                    word_dict = {word: weight for word, weight in words}
                    
                    # Create a word cloud
                    wordcloud = WordCloud(width=800, height=400, 
                                        background_color='white',
                                        max_words=50).generate_from_frequencies(word_dict)
                    
                    # Plot the word cloud
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f"Topic {topic_id}", fontsize=14)
                    axes[i].axis('off')
        
        # Hide any unused axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig('topic_wordclouds.png', dpi=300)
        plt.close()
        
    # 3. Topic-Document Heatmap
    # Create a matrix of documents vs topics
    try:
        topic_model.visualize_documents(filtered_df[filtered_df['Topic'] != -1]['News Content'].tolist())
        pio.write_html(topic_model.visualize_documents(), "topic_document_map.html")
        print("Topic-document visualization saved as 'topic_document_map.html'")
    except Exception as e:
        print(f"Error creating topic-document visualization: {e}")
    
    # 4. Topic Similarity
    try:
        similarity_matrix = topic_model.visualize_hierarchy()
        pio.write_html(similarity_matrix, "topic_hierarchy.html")
        print("Topic hierarchy visualization saved as 'topic_hierarchy.html'")
    except Exception as e:
        print(f"Error creating topic similarity visualization: {e}")
    
    # 5. Top Articles per Topic
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    
    y_pos = 0.98
    plt.text(0.5, y_pos, "Top Articles per Topic", fontsize=16, fontweight='bold', ha='center')
    y_pos -= 0.04
    
    # Get the most representative articles for each topic
    for topic_id in relevant_topics['Topic'][:3]:  # Limit to top 3 topics
        # Get topic keywords
        words = topic_model.get_topic(topic_id)
        top_words = [word for word, _ in words[:5]]
        
        plt.text(0.02, y_pos, f"Topic {topic_id}: {', '.join(top_words)}", fontsize=14, fontweight='bold')
        y_pos -= 0.03
        
        # Find articles with this topic
        topic_articles = filtered_df[filtered_df['Topic'] == topic_id]
        
        # Sort by topic probability if available
        if 'Topic_Probability' in topic_articles.columns:
            topic_articles = topic_articles.sort_values('Topic_Probability', ascending=False)
            
        # Get top 3 articles
        for i, (_, article) in enumerate(topic_articles.head(3).iterrows()):
            headline = article['Headline']
            
            # Truncate long headlines
            if len(headline) > 80:
                headline = headline[:77] + '...'
                
            plt.text(0.05, y_pos, f"{i+1}. {headline}", fontsize=12)
            y_pos -= 0.025
            
        y_pos -= 0.02  # Add space between topics
    
    plt.tight_layout()
    plt.savefig('top_articles_per_topic.png', dpi=300)
    plt.close()
    
    print("Topic modeling visualizations saved")

# Example usage
if __name__ == "__main__":
    # Load the preprocessed data with sentiment analysis
    try:
        input_csv = "sentiment_analyzed_articles.csv"
        df = pd.read_csv(input_csv)
    except:
        # Fall back to original processed data
        try:
            input_csv = "summarized_articles.csv"
            df = pd.read_csv(input_csv)
        except:
            input_csv = "indian_express_processed.csv"
            df = pd.read_csv(input_csv)
    
    print(f"Loaded data from {input_csv}")
    
    # Perform topic modeling on the news content
    topic_df, topic_model = perform_topic_modeling(df, text_column='News Content', n_topics=5)
    
    if topic_model is not None:
        # Visualize the topic modeling results
        visualize_topic_modeling_results(topic_df, topic_model)
        
        # Save the results
        topic_df.to_csv("topic_modeled_articles.csv", index=False)
        print("Topic modeled articles saved to 'topic_modeled_articles.csv'")
    else:
        print("Topic modeling was not successful")
    
    # Display sample topic results
    if 'Topic' in topic_df.columns:
        print("\nSample Topic Modeling Results:")
        for i in range(min(3, len(topic_df))):
            sample = topic_df.iloc[i]
            if pd.notna(sample['Topic']):
                print(f"\nArticle {i+1}:")
                print(f"Headline: {sample['Headline']}")
                print(f"Topic: {int(sample['Topic'])}")
                if 'Topic_Keywords' in topic_df.columns and pd.notna(sample['Topic_Keywords']):
                    print(f"Keywords: {sample['Topic_Keywords']}")