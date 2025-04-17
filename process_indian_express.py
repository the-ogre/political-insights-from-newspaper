import pandas as pd
import csv
import re
from datetime import datetime

def preprocess_indian_express_data(file_path):
    """
    Preprocess and convert the raw Indian Express data file to a clean CSV format.
    
    Args:
        file_path (str): Path to the raw data file
        
    Returns:
        pd.DataFrame: A clean dataframe with the processed data
    """
    # Read the raw file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find where the header ends and content begins
    header_match = re.search(r'Headline,Link,Written By,Timestamp,News Content\r?\n', content)
    if not header_match:
        raise ValueError("Could not find the header row in the file")
    
    header_end = header_match.end()
    header = content[:header_end].strip()
    content_part = content[header_end:]
    
    # Split the content into articles
    # This is a simplistic approach - regex pattern to match headline pattern
    article_pattern = r'"([^"]+)",([^,]+),([^,]*),(".*?IST"),(".*?")(?=\r?\n"|$)'
    articles = re.findall(article_pattern, content_part, re.DOTALL)
    
    if not articles:
        print("No articles found with the pattern. Trying an alternative approach...")
        # Try an alternative approach to extract articles
        lines = content.split('\n')
        header = lines[0]
        
        # Manual parsing approach
        articles = []
        current_article = []
        
        for line in lines[1:]:
            if line.startswith('"') and not current_article:
                current_article.append(line)
            elif current_article:
                current_article[-1] += '\n' + line
                if line.endswith('"') and '"read more"' not in line:
                    articles.append('\n'.join(current_article))
                    current_article = []
    
    # Process matched articles
    processed_articles = []
    for match in articles:
        try:
            if isinstance(match, tuple):
                headline, link, author, timestamp, news_content = match
            else:
                # Handle alternative parsing result
                parts = match.split(',')
                # This is a simplistic approach and might need refinement
                headline = parts[0]
                link = parts[1]
                author = parts[2] if len(parts) > 2 else ""
                timestamp = parts[3] if len(parts) > 3 else ""
                news_content = ','.join(parts[4:]) if len(parts) > 4 else ""
            
            # Clean the extracted fields
            headline = headline.strip('"')
            link = link.strip('"')
            author = author.strip('"')
            timestamp = timestamp.strip('"')
            news_content = news_content.strip('"')
            
            # Parse timestamp
            try:
                # Example: "August 9, 2023 21:03 IST"
                timestamp_clean = timestamp.replace('"', '')
                datetime_obj = datetime.strptime(timestamp_clean, "%B %d, %Y %H:%M IST")
                parsed_timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"Error parsing timestamp '{timestamp}': {e}")
                parsed_timestamp = timestamp
            
            processed_articles.append({
                'Headline': headline,
                'Link': link,
                'Written By': author,
                'Timestamp': parsed_timestamp,
                'News Content': news_content
            })
        except Exception as e:
            print(f"Error processing article: {e}")
            continue
    
    # Create a DataFrame
    df = pd.DataFrame(processed_articles)
    
    print(f"Successfully processed {len(df)} articles")
    return df

# Example usage
if __name__ == "__main__":
    file_path = "paste.txt"  # Change to your file path
    df = preprocess_indian_express_data(file_path)
    
    # Save to CSV
    output_path = "indian_express_processed.csv"
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Processed data saved to {output_path}")
    
    # Display the first few rows
    print("\nFirst few rows of the processed data:")
    print(df.head())