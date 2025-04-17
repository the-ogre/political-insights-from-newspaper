import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import json
from matplotlib.ticker import MaxNLocator

def analyze_weekly_sentiment(df):
    """
    Analyze sentiment trends by week.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data and timestamps
        
    Returns:
        pd.DataFrame: DataFrame with aggregated weekly sentiment data
    """
    print("Starting weekly sentiment analysis...")
    
    # Check if we have the required columns
    required_columns = ['Sentiment_Label', 'Timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_dtype(result_df['Timestamp']):
        try:
            # Try to parse the timestamp
            result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            
            # Try an alternative parsing approach
            try:
                # Example: "August 9, 2023 21:03 IST"
                result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'], format="%B %d, %Y %H:%M IST")
            except Exception as e:
                print(f"Alternative timestamp parsing failed: {e}")
                return None
    
    # Extract week number and year
    result_df['Year'] = result_df['Timestamp'].dt.isocalendar().year
    result_df['Week'] = result_df['Timestamp'].dt.isocalendar().week
    
    # Create a week identifier (Year-Week)
    result_df['Week_ID'] = result_df['Year'].astype(str) + '-' + result_df['Week'].astype(str).str.zfill(2)
    
    # Calculate start date of each week (Monday)
    result_df['Week_Start'] = result_df.apply(
        lambda x: datetime.fromisocalendar(x['Year'], x['Week'], 1), axis=1)
    
    # Convert sentiment labels to numeric values
    if 'Sentiment_Label' in result_df.columns:
        # Check the format of sentiment labels
        unique_labels = result_df['Sentiment_Label'].unique()
        
        if 'POSITIVE' in unique_labels and 'NEGATIVE' in unique_labels:
            result_df['Sentiment_Value'] = result_df['Sentiment_Label'].map(
                {'POSITIVE': 1, 'NEGATIVE': 0})
        elif 'Positive' in unique_labels and 'Negative' in unique_labels:
            result_df['Sentiment_Value'] = result_df['Sentiment_Label'].map(
                {'Positive': 1, 'Negative': 0, 'Neutral': 0.5})
        else:
            # If sentiment is stored as JSON
            if 'Sentiment_JSON' in result_df.columns:
                try:
                    result_df['Sentiment_Value'] = result_df['Sentiment_JSON'].apply(
                        lambda x: 1 if json.loads(x)['label'] == 'POSITIVE' else 0)
                except:
                    # Fallback to sentiment score if available
                    if 'Sentiment_Score' in result_df.columns:
                        result_df['Sentiment_Value'] = result_df['Sentiment_Score']
                    else:
                        print("Could not determine sentiment values")
                        return None
            else:
                # Fallback to sentiment score if available
                if 'Sentiment_Score' in result_df.columns:
                    result_df['Sentiment_Value'] = result_df['Sentiment_Score']
                else:
                    print("Could not determine sentiment values")
                    return None
    
    # Aggregate by week
    weekly_sentiment = result_df.groupby(['Week_ID', 'Week_Start']).agg(
        Positive_Count=('Sentiment_Value', lambda x: sum(x > 0.5)),
        Negative_Count=('Sentiment_Value', lambda x: sum(x <= 0.5)),
        Positive_Ratio=('Sentiment_Value', lambda x: sum(x > 0.5) / len(x)),
        Average_Sentiment=('Sentiment_Value', 'mean'),
        Article_Count=('Headline', 'count')
    ).reset_index()
    
    # Sort by week start date
    weekly_sentiment = weekly_sentiment.sort_values('Week_Start')
    
    print(f"Weekly sentiment analysis complete. Found {len(weekly_sentiment)} weeks of data.")
    
    return weekly_sentiment

def visualize_weekly_sentiment(weekly_sentiment):
    """
    Create visualizations for weekly sentiment trends.
    
    Args:
        weekly_sentiment (pd.DataFrame): DataFrame with weekly sentiment data
        
    Returns:
        None: Saves visualization files to disk
    """
    if weekly_sentiment is None or len(weekly_sentiment) == 0:
        print("No valid data for weekly sentiment visualization")
        return
    
    print("Generating weekly sentiment visualizations...")
    
    # Format week labels for better readability
    week_labels = weekly_sentiment['Week_Start'].dt.strftime('%b %d').tolist()
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 15))
    
    # 1. Positive Sentiment Ratio Trend
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    
    ax1.plot(week_labels, weekly_sentiment['Positive_Ratio'], 
             marker='o', markersize=10, linewidth=2.5, color='green')
    
    # Add data points
    for i, ratio in enumerate(weekly_sentiment['Positive_Ratio']):
        ax1.annotate(f"{ratio:.2f}", 
                   (i, ratio), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add a horizontal line at 0.5 for reference
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Week Starting', fontsize=12)
    ax1.set_ylabel('Positive Sentiment Ratio', fontsize=12)
    ax1.set_title('Weekly Trend of Positive Sentiment', fontsize=16)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Article Count by Week
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    
    bars = ax2.bar(week_labels, weekly_sentiment['Article_Count'], color='royalblue', alpha=0.7)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{int(height)}", 
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10)
    
    ax2.set_xlabel('Week Starting', fontsize=12)
    ax2.set_ylabel('Number of Articles', fontsize=12)
    ax2.set_title('Weekly Article Count', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Positive vs Negative Count
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    
    width = 0.4
    x = np.arange(len(week_labels))
    
    ax3.bar(x - width/2, weekly_sentiment['Positive_Count'], width, label='Positive', color='green', alpha=0.7)
    ax3.bar(x + width/2, weekly_sentiment['Negative_Count'], width, label='Negative', color='red', alpha=0.7)
    
    ax3.set_xlabel('Week Starting', fontsize=12)
    ax3.set_ylabel('Article Count', fontsize=12)
    ax3.set_title('Positive vs Negative Articles by Week', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(week_labels)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Average Sentiment Score
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    
    # Create a color gradient based on sentiment
    colors = ['red' if score < 0.5 else 'green' for score in weekly_sentiment['Average_Sentiment']]
    
    # Create the line plot with gradient colors
    for i in range(len(week_labels)-1):
        ax4.plot(week_labels[i:i+2], weekly_sentiment['Average_Sentiment'].iloc[i:i+2], 
               marker='o', linewidth=2.5, color=colors[i])
    
    # Add the final point if it's a single week
    if len(week_labels) == 1:
        ax4.plot(week_labels, weekly_sentiment['Average_Sentiment'], 
               marker='o', linewidth=2.5, color=colors[0])
    
    # Add data points
    for i, score in enumerate(weekly_sentiment['Average_Sentiment']):
        ax4.annotate(f"{score:.2f}", 
                   (i, score), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add a horizontal line at 0.5 for reference
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Week Starting', fontsize=12)
    ax4.set_ylabel('Average Sentiment Score', fontsize=12)
    ax4.set_title('Weekly Average Sentiment Score', fontsize=14)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('weekly_sentiment_analysis.png', dpi=300)
    plt.close()
    
    # Create a separate visualization that combines positive ratio and article count
    plt.figure(figsize=(12, 8))
    
    # Create primary y-axis for positive ratio
    ax_ratio = plt.gca()
    
    # Plot positive ratio
    line1 = ax_ratio.plot(week_labels, weekly_sentiment['Positive_Ratio'], 
                         marker='o', linewidth=2.5, color='green', label='Positive Ratio')
    
    ax_ratio.set_xlabel('Week Starting', fontsize=12)
    ax_ratio.set_ylabel('Positive Sentiment Ratio', fontsize=12, color='green')
    ax_ratio.set_ylim(0, 1)
    ax_ratio.tick_params(axis='y', labelcolor='green')
    
    # Create secondary y-axis for article count
    ax_count = ax_ratio.twinx()
    
    # Plot article count
    line2 = ax_count.bar(week_labels, weekly_sentiment['Article_Count'], 
                        alpha=0.3, color='blue', label='Article Count')
    
    ax_count.set_ylabel('Number of Articles', fontsize=12, color='blue')
    ax_count.tick_params(axis='y', labelcolor='blue')
    
    # Add title
    plt.title('Weekly Sentiment Ratio vs. Article Count', fontsize=16)
    
    # Add a legend
    lines = [line1[0], plt.Rectangle((0,0),1,1, color='blue', alpha=0.3)]
    labels = ['Positive Ratio', 'Article Count']
    plt.legend(lines, labels, loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax_ratio.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('weekly_sentiment_vs_count.png', dpi=300)
    plt.close()
    
    print("Weekly sentiment visualizations saved")

def analyze_entity_sentiment_by_week(df, target_entities=None):
    """
    Analyze sentiment trends for specific entities by week.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment and entity data
        target_entities (list): List of entities to analyze
        
    Returns:
        pd.DataFrame: DataFrame with entity sentiment by week
    """
    if target_entities is None:
        target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    print(f"Starting weekly entity sentiment analysis for: {', '.join(target_entities)}")
    
    # Check if we have the required columns
    required_columns = ['Sentiment_Value', 'Timestamp']
    entity_columns = [f'{entity}_Count' for entity in target_entities]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    missing_entities = [col for col in entity_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    if missing_entities:
        print(f"Missing entity columns: {missing_entities}")
        print("Entity sentiment analysis requires NER to be performed first")
        return None
    
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_dtype(result_df['Timestamp']):
        try:
            result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            return None
    
    # Extract week number and year
    result_df['Year'] = result_df['Timestamp'].dt.isocalendar().year
    result_df['Week'] = result_df['Timestamp'].dt.isocalendar().week
    result_df['Week_ID'] = result_df['Year'].astype(str) + '-' + result_df['Week'].astype(str).str.zfill(2)
    result_df['Week_Start'] = result_df.apply(
        lambda x: datetime.fromisocalendar(x['Year'], x['Week'], 1), axis=1)
    
    # Calculate entity sentiment for each article
    for entity in target_entities:
        entity_col = f'{entity}_Count'
        result_df[f'{entity}_Sentiment'] = result_df.apply(
            lambda x: x['Sentiment_Value'] if x[entity_col] > 0 else None, axis=1)
    
    # Aggregate by week
    weekly_entity_sentiment = []
    
    for week_id, week_group in result_df.groupby('Week_ID'):
        week_start = week_group['Week_Start'].iloc[0]
        week_data = {'Week_ID': week_id, 'Week_Start': week_start}
        
        for entity in target_entities:
            # Articles mentioning this entity
            entity_articles = week_group[week_group[f'{entity}_Count'] > 0]
            
            if len(entity_articles) > 0:
                # Average sentiment for this entity
                avg_sentiment = entity_articles['Sentiment_Value'].mean()
                # Entity mention count
                mention_count = entity_articles[f'{entity}_Count'].sum()
                # Positive ratio
                positive_ratio = (entity_articles['Sentiment_Value'] > 0.5).mean()
                
                week_data[f'{entity}_Avg_Sentiment'] = avg_sentiment
                week_data[f'{entity}_Mention_Count'] = mention_count
                week_data[f'{entity}_Positive_Ratio'] = positive_ratio
            else:
                week_data[f'{entity}_Avg_Sentiment'] = None
                week_data[f'{entity}_Mention_Count'] = 0
                week_data[f'{entity}_Positive_Ratio'] = None
        
        weekly_entity_sentiment.append(week_data)
    
    # Convert to DataFrame and sort by week
    weekly_entity_df = pd.DataFrame(weekly_entity_sentiment)
    weekly_entity_df = weekly_entity_df.sort_values('Week_Start')
    
    print(f"Weekly entity sentiment analysis complete for {len(weekly_entity_df)} weeks")
    
    return weekly_entity_df

def visualize_entity_sentiment_by_week(weekly_entity_df, target_entities=None):
    """
    Visualize weekly sentiment trends for specific entities.
    
    Args:
        weekly_entity_df (pd.DataFrame): DataFrame with entity sentiment by week
        target_entities (list): List of entities to analyze
        
    Returns:
        None: Saves visualization files to disk
    """
    if target_entities is None:
        target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    if weekly_entity_df is None or len(weekly_entity_df) == 0:
        print("No valid data for entity sentiment visualization")
        return
    
    print("Generating entity sentiment visualizations...")
    
    # Format week labels
    week_labels = weekly_entity_df['Week_Start'].dt.strftime('%b %d').tolist()
    
    # Entity colors
    entity_colors = {
        'BJP': '#FF9F40',           # Orange
        'Congress': '#36A2EB',      # Blue
        'Rahul Gandhi': '#4BC0C0',  # Teal
        'Narendra Modi': '#FF6384'  # Red
    }
    
    # 1. Entity Positive Sentiment Ratio by Week
    plt.figure(figsize=(12, 8))
    
    for entity in target_entities:
        positive_ratio_col = f'{entity}_Positive_Ratio'
        
        if positive_ratio_col in weekly_entity_df.columns:
            # Filter out None values
            valid_weeks = ~weekly_entity_df[positive_ratio_col].isna()
            
            if valid_weeks.any():
                plt.plot(
                    [week_labels[i] for i in range(len(week_labels)) if valid_weeks.iloc[i]],
                    weekly_entity_df.loc[valid_weeks, positive_ratio_col],
                    marker='o',
                    linewidth=2,
                    label=entity,
                    color=entity_colors.get(entity, None)
                )
    
    plt.xlabel('Week Starting', fontsize=12)
    plt.ylabel('Positive Sentiment Ratio', fontsize=12)
    plt.title('Weekly Positive Sentiment Ratio by Entity', fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('weekly_entity_sentiment_ratio.png', dpi=300)
    plt.close()
    
    # 2. Entity Mention Count by Week
    plt.figure(figsize=(12, 8))
    
    width = 0.2
    x = np.arange(len(week_labels))
    
    for i, entity in enumerate(target_entities):
        mention_count_col = f'{entity}_Mention_Count'
        
        if mention_count_col in weekly_entity_df.columns:
            plt.bar(
                x + (i - len(target_entities)/2 + 0.5) * width,
                weekly_entity_df[mention_count_col],
                width,
                label=entity,
                color=entity_colors.get(entity, None)
            )
    
    plt.xlabel('Week Starting', fontsize=12)
    plt.ylabel('Mention Count', fontsize=12)
    plt.title('Weekly Entity Mention Count', fontsize=16)
    plt.xticks(x, week_labels, rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weekly_entity_mentions.png', dpi=300)
    plt.close()
    
    # 3. Entity Average Sentiment by Week
    plt.figure(figsize=(12, 8))
    
    for entity in target_entities:
        avg_sentiment_col = f'{entity}_Avg_Sentiment'
        
        if avg_sentiment_col in weekly_entity_df.columns:
            # Filter out None values
            valid_weeks = ~weekly_entity_df[avg_sentiment_col].isna()
            
            if valid_weeks.any():
                plt.plot(
                    [week_labels[i] for i in range(len(week_labels)) if valid_weeks.iloc[i]],
                    weekly_entity_df.loc[valid_weeks, avg_sentiment_col],
                    marker='o',
                    linewidth=2,
                    label=entity,
                    color=entity_colors.get(entity, None)
                )
    
    plt.xlabel('Week Starting', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.title('Weekly Average Sentiment Score by Entity', fontsize=16)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('weekly_entity_sentiment_score.png', dpi=300)
    plt.close()
    
    # 4. Combined visualization with all entities
    fig, axes = plt.subplots(len(target_entities), 1, figsize=(12, 4*len(target_entities)), sharex=True)
    
    for i, entity in enumerate(target_entities):
        ax = axes[i]
        
        mention_count_col = f'{entity}_Mention_Count'
        sentiment_col = f'{entity}_Avg_Sentiment'
        
        # Create primary y-axis for sentiment
        color = entity_colors.get(entity, None)
        
        valid_weeks = ~weekly_entity_df[sentiment_col].isna()
        
        if valid_weeks.any():
            # Plot sentiment line
            ax.plot(
                [week_labels[j] for j in range(len(week_labels)) if valid_weeks.iloc[j]],
                weekly_entity_df.loc[valid_weeks, sentiment_col],
                marker='o',
                linewidth=2,
                color=color,
                label='Sentiment'
            )
        
        ax.set_ylabel(f'Sentiment', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0, 1)
        
        # Add a horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Create secondary y-axis for mention count
        ax2 = ax.twinx()
        ax2.bar(week_labels, weekly_entity_df[mention_count_col], alpha=0.3, color=color)
        ax2.set_ylabel('Mentions', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Set entity name as subplot title
        ax.set_title(entity, fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Set x-axis label on the bottom subplot
    axes[-1].set_xlabel('Week Starting', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle('Entity Sentiment and Mention Count by Week', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('weekly_entity_combined.png', dpi=300)
    plt.close()
    
    print("Entity sentiment visualizations saved")

# Example usage
if __name__ == "__main__":
    # Load the processed data with NER
    try:
        input_csv = "ner_analyzed_articles.csv"
        df = pd.read_csv(input_csv)
    except:
        # Fall back to previous processing stages
        try:
            input_csv = "topic_modeled_articles.csv"
            df = pd.read_csv(input_csv)
        except:
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
    
    # If sentiment data is not available, try to perform sentiment analysis first
    if 'Sentiment_Label' not in df.columns and 'Sentiment_Score' not in df.columns:
        print("Sentiment data not found. Please run sentiment analysis first.")
    else:
        # Perform weekly sentiment analysis
        weekly_sentiment = analyze_weekly_sentiment(df)
        
        if weekly_sentiment is not None:
            # Visualize the weekly sentiment results
            visualize_weekly_sentiment(weekly_sentiment)
            
            # Save the results
            weekly_sentiment.to_csv("weekly_sentiment.csv", index=False)
            print("Weekly sentiment data saved to 'weekly_sentiment.csv'")
            
            # If entity data is available, perform entity sentiment analysis by week
            target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
            entity_columns = [f'{entity}_Count' for entity in target_entities]
            
            if all(col in df.columns for col in entity_columns):
                weekly_entity_sentiment = analyze_entity_sentiment_by_week(df, target_entities)
                
                if weekly_entity_sentiment is not None:
                    visualize_entity_sentiment_by_week(weekly_entity_sentiment, target_entities)
                    
                    # Save the results
                    weekly_entity_sentiment.to_csv("weekly_entity_sentiment.csv", index=False)
                    print("Weekly entity sentiment data saved to 'weekly_entity_sentiment.csv'")
            else:
                print("Entity data not found. Please run NER analysis first for entity-specific weekly sentiment.")
        
        # Display sample weekly sentiment results
        if weekly_sentiment is not None and len(weekly_sentiment) > 0:
            print("\nSample Weekly Sentiment Results:")
            for i in range(min(3, len(weekly_sentiment))):
                sample = weekly_sentiment.iloc[i]
                week_start = sample['Week_Start'].strftime('%B %d, %Y')
                print(f"\nWeek starting {week_start}:")
                print(f"  Articles: {int(sample['Article_Count'])}")
                print(f"  Positive ratio: {sample['Positive_Ratio']:.2f}")
                print(f"  Positive articles: {int(sample['Positive_Count'])}")
                print(f"  Negative articles: {int(sample['Negative_Count'])}")