import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import time
import spacy
import re
from collections import Counter
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def perform_ner_analysis(df, text_column='News Content', target_entities=None):
    """
    Extract named entities from news articles, focusing on specified target entities.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news articles
        text_column (str): Name of the column containing the text to analyze
        target_entities (list): List of specific entities to track (e.g., ['BJP', 'Congress'])
        
    Returns:
        tuple: (DataFrame with entity data, Dictionary of entity counts by article)
    """
    if target_entities is None:
        target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    print(f"Starting Named Entity Recognition analysis for: {', '.join(target_entities)}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    start_time = time.time()
    
    # For simplistic NER focused on specific entities, we'll use a regex approach for demonstration
    # In a real implementation, you'd want to use a proper NER model, but this is faster for specific targets
    
    # Prepare regex patterns for target entities
    patterns = {}
    for entity in target_entities:
        # Create case-insensitive pattern
        patterns[entity] = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        
        # Add some common variations
        if entity == 'BJP':
            patterns['BJP'] = re.compile(r'\b(BJP|Bharatiya Janata Party)\b', re.IGNORECASE)
        elif entity == 'Congress':
            patterns['Congress'] = re.compile(r'\b(Congress|INC|Indian National Congress)\b', re.IGNORECASE)
        elif entity == 'Rahul Gandhi':
            patterns['Rahul Gandhi'] = re.compile(r'\b(Rahul Gandhi|Rahul|Gandhi)\b', re.IGNORECASE)
        elif entity == 'Narendra Modi':
            patterns['Narendra Modi'] = re.compile(r'\b(Narendra Modi|Modi|PM Modi|Prime Minister Modi)\b', re.IGNORECASE)
    
    # Initialize entity counts for each article
    article_entity_counts = []
    
    # Process each article
    for idx, row in result_df.iterrows():
        text = row[text_column]
        
        # Skip empty texts
        if not isinstance(text, str) or len(text) == 0:
            article_entity_counts.append({entity: 0 for entity in target_entities})
            continue
        
        # Count occurrences of each target entity
        entity_counts = {}
        for entity, pattern in patterns.items():
            matches = pattern.findall(text)
            entity_counts[entity] = len(matches)
        
        article_entity_counts.append(entity_counts)
    
    # Add entity counts to the DataFrame
    for entity in target_entities:
        result_df[f'{entity}_Count'] = [counts[entity] for counts in article_entity_counts]
    
    # Add total entity count
    result_df['Total_Entity_Count'] = result_df[[f'{entity}_Count' for entity in target_entities]].sum(axis=1)
    
    end_time = time.time()
    print(f"NER analysis completed in {end_time - start_time:.2f} seconds")
    
    # Calculate total entity counts across all articles
    total_entity_counts = {entity: sum(counts[entity] for counts in article_entity_counts) 
                         for entity in target_entities}
    
    print("Entity mention counts across all articles:")
    for entity, count in total_entity_counts.items():
        print(f"{entity}: {count}")
    
    return result_df, article_entity_counts, total_entity_counts

def visualize_ner_results(df, total_entity_counts, target_entities=None):
    """
    Create visualizations for the NER analysis results.
    
    Args:
        df (pd.DataFrame): DataFrame with entity count data
        total_entity_counts (dict): Dictionary of total entity counts
        target_entities (list): List of target entities
    
    Returns:
        None: Saves visualization files to disk
    """
    if target_entities is None:
        target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    print("Generating NER analysis visualizations...")
    
    # 1. Overall Entity Frequency
    plt.figure(figsize=(12, 6))
    
    entities = list(total_entity_counts.keys())
    counts = list(total_entity_counts.values())
    
    # Custom colors for entities
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]  # Descending order
    sorted_entities = [entities[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars = plt.bar(range(len(sorted_entities)), sorted_counts, color=sorted_colors)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Political Entity', fontsize=14)
    plt.ylabel('Mention Count', fontsize=14)
    plt.title('Key Political Entity Mentions in News Articles', fontsize=16)
    plt.xticks(range(len(sorted_entities)), sorted_entities, rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entity_counts.png', dpi=300)
    plt.close()
    
    # 2. Entity Mentions by Article
    plt.figure(figsize=(15, 8))
    
    # Prepare data for stacked bar chart
    article_indices = range(len(df))
    entity_data = {entity: df[f'{entity}_Count'].values for entity in target_entities}
    
    bottom = np.zeros(len(df))
    
    for i, entity in enumerate(target_entities):
        plt.bar(article_indices, entity_data[entity], bottom=bottom, label=entity, alpha=0.7, color=colors[i % len(colors)])
        bottom += entity_data[entity]
    
    plt.xlabel('Article Index', fontsize=14)
    plt.ylabel('Number of Mentions', fontsize=14)
    plt.title('Political Entity Mentions by Article', fontsize=16)
    plt.legend(title='Entity')
    plt.xticks(article_indices)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entity_mentions_by_article.png', dpi=300)
    plt.close()
    
    # 3. Entity Co-occurrence Heatmap
    plt.figure(figsize=(10, 8))
    
    # Calculate co-occurrence
    co_occurrence = np.zeros((len(target_entities), len(target_entities)))
    
    for i, entity1 in enumerate(target_entities):
        for j, entity2 in enumerate(target_entities):
            if i == j:
                co_occurrence[i, j] = total_entity_counts[entity1]
            else:
                # Count articles where both entities are mentioned
                co_occur_count = ((df[f'{entity1}_Count'] > 0) & (df[f'{entity2}_Count'] > 0)).sum()
                co_occurrence[i, j] = co_occur_count
    
    # Create the heatmap
    sns.heatmap(co_occurrence, annot=True, fmt='g', cmap='YlGnBu',
               xticklabels=target_entities, yticklabels=target_entities)
    
    plt.title('Entity Co-occurrence Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('entity_cooccurrence.png', dpi=300)
    plt.close()
    
    # 4. Entity Relationship Network
    plt.figure(figsize=(12, 10))
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for entity in target_entities:
        G.add_node(entity, count=total_entity_counts[entity])
    
    # Add edges
    for i, entity1 in enumerate(target_entities):
        for j, entity2 in enumerate(target_entities):
            if i < j:  # Avoid duplicate edges
                co_occur_count = co_occurrence[i, j]
                if co_occur_count > 0:
                    G.add_edge(entity1, entity2, weight=co_occur_count)
    
    # Node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Node sizes based on entity counts
    node_sizes = [total_entity_counts[entity] * 20 for entity in G.nodes()]
    
    # Edge weights based on co-occurrence
    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors[:len(G.nodes())], alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title('Entity Relationship Network', fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('entity_network.png', dpi=300)
    plt.close()
    
    # 5. Entity Distribution with Headlines
    plt.figure(figsize=(15, 8))
    plt.axis('off')
    
    # Create a colormap
    cmap = plt.cm.viridis
    
    y_pos = 0.95
    plt.text(0.5, y_pos, "Top Articles by Entity Mentions", fontsize=18, fontweight='bold', ha='center')
    y_pos -= 0.05
    
    # For each entity, find the top 2 articles
    for i, entity in enumerate(target_entities):
        entity_color = cmap(i / len(target_entities))
        
        plt.text(0.02, y_pos, f"{entity}", fontsize=16, fontweight='bold', color=entity_color)
        y_pos -= 0.03
        
        # Sort articles by this entity count
        top_articles = df.sort_values(f'{entity}_Count', ascending=False).head(2)
        
        for j, (_, article) in enumerate(top_articles.iterrows()):
            headline = article['Headline']
            count = article[f'{entity}_Count']
            
            # Truncate long headlines
            if len(headline) > 80:
                headline = headline[:77] + '...'
                
            plt.text(0.05, y_pos, f"{headline}", fontsize=12)
            plt.text(0.9, y_pos, f"({count} mentions)", fontsize=12, color=entity_color)
            y_pos -= 0.025
        
        y_pos -= 0.02  # Add space between entities
    
    plt.tight_layout()
    plt.savefig('top_articles_by_entity.png', dpi=300)
    plt.close()
    
    print("NER analysis visualizations saved")

def perform_advanced_ner_with_model(df, text_column='News Content', target_entities=None):
    """
    Perform advanced NER using a BERT-based model from Hugging Face.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news articles
        text_column (str): Name of the column containing the text to analyze
        target_entities (list): List of specific entities to track
        
    Returns:
        pd.DataFrame: DataFrame with entity data from the model
    """
    if target_entities is None:
        target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    print(f"Starting advanced NER with BERT model...")
    
    try:
        # Initialize NER pipeline
        ner = pipeline("ner", model="dslim/bert-base-NER")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Initialize entity counts
        for entity in target_entities:
            result_df[f'{entity}_NER_Count'] = 0
        
        # Process each article
        for idx, row in result_df.iterrows():
            text = row[text_column]
            
            # Skip empty texts
            if not isinstance(text, str) or len(text) == 0:
                continue
            
            try:
                # Process text in chunks due to token limits
                chunk_size = 400
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                
                entity_counts = {entity: 0 for entity in target_entities}
                
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
                    for entity_info in grouped_entities:
                        entity_text = entity_info['word'].lower()
                        
                        # Check for target entities
                        for target in target_entities:
                            if target.lower() in entity_text:
                                entity_counts[target] += 1
                                break
                
                # Update the DataFrame
                for entity, count in entity_counts.items():
                    result_df.at[idx, f'{entity}_NER_Count'] = count
                
                print(f"Processed article {idx+1}/{len(result_df)} with advanced NER")
                
            except Exception as e:
                print(f"Error in advanced NER for article {idx}: {e}")
        
        return result_df
        
    except Exception as e:
        print(f"Error initializing advanced NER model: {e}")
        print("Falling back to regex-based NER")
        return df

# Example usage
if __name__ == "__main__":
    # Load the processed data with topic modeling
    try:
        input_csv = "topic_modeled_articles.csv"
        df = pd.read_csv(input_csv)
    except:
        # Fall back to previous processing stage
        try:
            input_csv = "sentiment_analyzed_articles.csv"
            df = pd.read_csv(input_csv)
        except:
            try:
                input_csv = "summarized_articles.csv"
                df = pd.read_csv(input_csv)
            except:
                input_csv = "indian_express_processed.csv"
                df = pd.read_csv(input_csv)
    
    print(f"Loaded data from {input_csv}")
    
    # Define target entities
    target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    # Perform NER analysis
    ner_df, article_entity_counts, total_entity_counts = perform_ner_analysis(
        df, text_column='News Content', target_entities=target_entities)
    
    # Visualize the NER results
    visualize_ner_results(ner_df, total_entity_counts, target_entities)
    
    # Try advanced NER with a BERT model if possible
    try:
        advanced_ner_df = perform_advanced_ner_with_model(
            ner_df, text_column='News Content', target_entities=target_entities)
        
        # Save the results
        advanced_ner_df.to_csv("ner_analyzed_articles.csv", index=False)
        print("NER analyzed articles saved to 'ner_analyzed_articles.csv'")
    except Exception as e:
        print(f"Advanced NER not available: {e}")
        # Save the results from the basic NER
        ner_df.to_csv("ner_analyzed_articles.csv", index=False)
        print("NER analyzed articles saved to 'ner_analyzed_articles.csv'")
    
    # Display sample NER results
    print("\nSample NER Results:")
    for i in range(min(3, len(ner_df))):
        sample = ner_df.iloc[i]
        print(f"\nArticle {i+1}: {sample['Headline']}")
        for entity in target_entities:
            count = sample[f'{entity}_Count']
            print(f"  {entity}: {count} mentions")