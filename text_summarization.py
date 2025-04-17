import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

def perform_text_summarization(df, model_name="facebook/bart-large-cnn", max_length=150, min_length=40):
    """
    Summarize news articles using a BERT-based model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the news articles
        model_name (str): Name of the Hugging Face model to use
        max_length (int): Maximum length of the summary in tokens
        min_length (int): Minimum length of the summary in tokens
        
    Returns:
        pd.DataFrame: DataFrame with an additional 'Summary' column
    """
    print(f"Initializing summarization model: {model_name}")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add a summary column
    result_df['Summary'] = None
    result_df['Summary_Length'] = None
    result_df['Original_Length'] = None
    result_df['Compression_Ratio'] = None
    
    print("Starting text summarization...")
    start_time = time.time()
    
    # Process each article
    for idx, row in result_df.iterrows():
        text = row['News Content']
        
        # Skip empty or very short texts
        if not isinstance(text, str) or len(text) < 100:
            result_df.at[idx, 'Summary'] = "Text too short to summarize."
            continue
        
        try:
            # Most models have a token limit (e.g., 1024 tokens for BART)
            # Truncate text if necessary by taking the first portion
            if len(text) > 4096:
                text = text[:4096]
            
            # Generate summary
            summary = summarizer(text, max_length=max_length, min_length=min_length, 
                               do_sample=False)[0]['summary_text']
            
            # Store the summary and length metrics
            result_df.at[idx, 'Summary'] = summary
            result_df.at[idx, 'Summary_Length'] = len(summary)
            result_df.at[idx, 'Original_Length'] = len(text)
            result_df.at[idx, 'Compression_Ratio'] = len(summary) / len(text) if len(text) > 0 else 0
            
            print(f"Summarized article {idx+1}/{len(result_df)}")
        except Exception as e:
            print(f"Error summarizing article {idx}: {e}")
            result_df.at[idx, 'Summary'] = f"Error: {str(e)}"
    
    end_time = time.time()
    print(f"Text summarization completed in {end_time - start_time:.2f} seconds")
    
    return result_df

def visualize_summarization_results(df):
    """
    Create visualizations for the text summarization results.
    
    Args:
        df (pd.DataFrame): DataFrame with summary information
    
    Returns:
        None: Saves visualization files to disk
    """
    print("Generating text summarization visualizations...")
    
    # Filter out rows with missing data
    filtered_df = df.dropna(subset=['Summary_Length', 'Original_Length', 'Compression_Ratio'])
    
    if len(filtered_df) == 0:
        print("No valid summarization data to visualize")
        return
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Length comparison
    plt.subplot(2, 2, 1)
    indices = range(len(filtered_df))
    
    plt.bar(indices, filtered_df['Original_Length'], alpha=0.7, label='Original Text')
    plt.bar(indices, filtered_df['Summary_Length'], alpha=0.7, label='Summary')
    
    plt.xlabel('Article Index')
    plt.ylabel('Character Count')
    plt.title('Original Text vs Summary Length')
    plt.legend()
    plt.xticks(indices)
    
    # Plot 2: Compression ratio
    plt.subplot(2, 2, 2)
    plt.bar(indices, filtered_df['Compression_Ratio'] * 100, color='green')
    
    plt.xlabel('Article Index')
    plt.ylabel('Compression Ratio (%)')
    plt.title('Text Compression Ratio (Lower is Better)')
    plt.xticks(indices)
    
    # Plot 3: Length distribution
    plt.subplot(2, 2, 3)
    
    # Create a DataFrame for easier plotting
    length_df = pd.DataFrame({
        'Original': filtered_df['Original_Length'],
        'Summary': filtered_df['Summary_Length']
    })
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(length_df, var_name='Type', value_name='Length')
    
    # Create the box plot
    sns.boxplot(x='Type', y='Length', data=melted_df)
    plt.title('Length Distribution')
    plt.ylabel('Character Count')
    
    # Plot 4: Summary sample
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Display a sample summary
    sample_idx = 0 if len(filtered_df) > 0 else 0
    if sample_idx < len(filtered_df):
        headline = filtered_df.iloc[sample_idx]['Headline']
        summary = filtered_df.iloc[sample_idx]['Summary']
        
        plt.text(0, 0.9, "Sample Summary", fontsize=14, fontweight='bold')
        plt.text(0, 0.8, f"Headline: {headline[:50]}...", fontsize=10)
        plt.text(0, 0.6, "Summary:", fontsize=12)
        
        # Wrap the summary text
        summary_lines = []
        line = ""
        for word in summary.split():
            if len(line + " " + word) <= 80:
                line += " " + word
            else:
                summary_lines.append(line)
                line = word
        if line:
            summary_lines.append(line)
        
        for i, line in enumerate(summary_lines):
            plt.text(0, 0.5 - (i * 0.05), line, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('text_summarization_analysis.png', dpi=300)
    plt.close()
    
    print("Text summarization visualization saved as 'text_summarization_analysis.png'")

# Example usage
if __name__ == "__main__":
    # Load the preprocessed data
    input_csv = "indian_express_processed.csv"
    df = pd.read_csv(input_csv)
    
    # Perform text summarization
    summarized_df = perform_text_summarization(df)
    
    # Visualize the results
    visualize_summarization_results(summarized_df)
    
    # Save the results
    summarized_df.to_csv("summarized_articles.csv", index=False)
    print("Summarized articles saved to 'summarized_articles.csv'")
    
    # Display a sample summary
    if len(summarized_df) > 0:
        print("\nSample Summary:")
        sample = summarized_df.iloc[0]
        print(f"Headline: {sample['Headline']}")
        print(f"Original Length: {sample['Original_Length']} characters")
        print(f"Summary Length: {sample['Summary_Length']} characters")
        print(f"Compression Ratio: {sample['Compression_Ratio']*100:.2f}%")
        print(f"Summary: {sample['Summary']}")