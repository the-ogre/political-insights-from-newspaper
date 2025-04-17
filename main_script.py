import os
import pandas as pd
import argparse
from datetime import datetime

# Import all our modules
from process_indian_express import preprocess_indian_express_data
from text_summarization import perform_text_summarization, visualize_summarization_results
from sentiment_analysis import perform_sentiment_analysis, visualize_sentiment_results
from topic_modeling import perform_topic_modeling, visualize_topic_modeling_results
from named_entity_recognition import perform_ner_analysis, visualize_ner_results
from weekly_sentiment_analysis import analyze_weekly_sentiment, visualize_weekly_sentiment, analyze_entity_sentiment_by_week, visualize_entity_sentiment_by_week

def setup_output_directory():
    """Create a timestamped output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main(input_file, output_dir=None, tasks=None):
    """
    Run the complete NLP analysis pipeline on the Indian Express dataset.
    
    Args:
        input_file (str): Path to the input file
        output_dir (str): Path to the output directory
        tasks (list): Specific tasks to run, or None for all tasks
    """
    if output_dir is None:
        output_dir = setup_output_directory()
    
    print(f"\n{'='*50}")
    print(f"Starting Indian Express News Analysis Pipeline")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    if tasks:
        print(f"Selected tasks: {', '.join(tasks)}")
    print(f"{'='*50}\n")
    
    # Define target entities for NER
    target_entities = ['BJP', 'Congress', 'Rahul Gandhi', 'Narendra Modi']
    
    # Step 1: Preprocess the data
    print("\n--- Step 1: Data Preprocessing ---")
    df = preprocess_indian_express_data(input_file)
    preprocessed_file = os.path.join(output_dir, "01_preprocessed_data.csv")
    df.to_csv(preprocessed_file, index=False)
    print(f"Preprocessed data saved to {preprocessed_file}")
    
    # Step 2: Text Summarization
    if tasks is None or 'summarization' in tasks:
        print("\n--- Step 2: Text Summarization ---")
        summarized_df = perform_text_summarization(df)
        summarized_file = os.path.join(output_dir, "02_summarized_articles.csv")
        summarized_df.to_csv(summarized_file, index=False)
        print(f"Summarized articles saved to {summarized_file}")
        
        # Visualize the summarization results
        visualize_summarization_results(summarized_df)
        
        # Use the summarized data for subsequent steps
        df = summarized_df
    
    # Step 3: Sentiment Analysis
    if tasks is None or 'sentiment' in tasks:
        print("\n--- Step 3: Sentiment Analysis ---")
        # Use either the summary or original content based on availability
        text_column = 'Summary' if 'Summary' in df.columns else 'News Content'
        sentiment_df = perform_sentiment_analysis(df, text_column=text_column)
        sentiment_file = os.path.join(output_dir, "03_sentiment_analyzed_articles.csv")
        sentiment_df.to_csv(sentiment_file, index=False)
        print(f"Sentiment analyzed articles saved to {sentiment_file}")
        
        # Visualize the sentiment results
        visualize_sentiment_results(sentiment_df)
        
        # Use the sentiment-analyzed data for subsequent steps
        df = sentiment_df
    
    # Step 4: Topic Modeling
    if tasks is None or 'topics' in tasks:
        print("\n--- Step 4: Topic Modeling ---")
        # Use the original news content for topic modeling, not summaries
        text_column = 'News Content'
        topic_df, topic_model = perform_topic_modeling(df, text_column=text_column, n_topics=5)
        
        if topic_model is not None:
            topic_file = os.path.join(output_dir, "04_topic_modeled_articles.csv")
            topic_df.to_csv(topic_file, index=False)
            print(f"Topic modeled articles saved to {topic_file}")
            
            # Visualize the topic modeling results
            visualize_topic_modeling_results(topic_df, topic_model)
            
            # Use the topic-modeled data for subsequent steps
            df = topic_df
    
    # Step 5: Named Entity Recognition
    if tasks is None or 'ner' in tasks:
        print("\n--- Step 5: Named Entity Recognition ---")
        text_column = 'News Content'  # Always use full content for NER
        ner_df, article_entity_counts, total_entity_counts = perform_ner_analysis(
            df, text_column=text_column, target_entities=target_entities)
        
        ner_file = os.path.join(output_dir, "05_ner_analyzed_articles.csv")
        ner_df.to_csv(ner_file, index=False)
        print(f"NER analyzed articles saved to {ner_file}")
        
        # Visualize the NER results
        visualize_ner_results(ner_df, total_entity_counts, target_entities)
        
        # Use the NER-analyzed data for subsequent steps
        df = ner_df
    
    # Step 6: Weekly Sentiment Analysis
    if tasks is None or 'weekly' in tasks:
        print("\n--- Step 6: Weekly Sentiment Analysis ---")
        weekly_sentiment = analyze_weekly_sentiment(df)
        
        if weekly_sentiment is not None:
            weekly_file = os.path.join(output_dir, "06_weekly_sentiment.csv")
            weekly_sentiment.to_csv(weekly_file, index=False)
            print(f"Weekly sentiment data saved to {weekly_file}")
            
            # Visualize the weekly sentiment results
            visualize_weekly_sentiment(weekly_sentiment)
            
            # If entity data is available, perform entity sentiment analysis by week
            entity_columns = [f'{entity}_Count' for entity in target_entities]
            
            if all(col in df.columns for col in entity_columns):
                weekly_entity_sentiment = analyze_entity_sentiment_by_week(df, target_entities)
                
                if weekly_entity_sentiment is not None:
                    weekly_entity_file = os.path.join(output_dir, "06_weekly_entity_sentiment.csv")
                    weekly_entity_sentiment.to_csv(weekly_entity_file, index=False)
                    print(f"Weekly entity sentiment data saved to {weekly_entity_file}")
                    
                    # Visualize the entity sentiment by week
                    visualize_entity_sentiment_by_week(weekly_entity_sentiment, target_entities)
    
    # Move all visualization files to the output directory
    import glob
    for img_file in glob.glob("*.png"):
        os.rename(img_file, os.path.join(output_dir, img_file))
    
    for html_file in glob.glob("*.html"):
        os.rename(html_file, os.path.join(output_dir, html_file))
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! All results saved to {output_dir}")
    print(f"{'='*50}\n")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indian Express News Analysis Pipeline')
    parser.add_argument('input_file', help='Path to the input file')
    parser.add_argument('--output-dir', help='Path to the output directory')
    parser.add_argument('--tasks', nargs='+', choices=['summarization', 'sentiment', 'topics', 'ner', 'weekly'],
                        help='Specific tasks to run (default: all)')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_dir, args.tasks)