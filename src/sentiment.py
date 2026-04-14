import pandas as pd
from textblob import TextBlob

# If the generated mock data already has a 'Sentiment' field, we can use that, 
# but here is the NLP block fulfilling the TextBlob requirement.
def analyze_sentiment(text):
    '''Returns polarity score from -1.0 to 1.0 using TextBlob'''
    try:
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity
    except:
        return 0.0

def process_social_media(filepath):
    '''Loads social media data, applies sentiment NLP, and groups by city'''
    df = pd.read_csv(filepath)
    
    # Run NLP on raw text (even though mock might have a simulated Sentiment column, 
    # doing it via TextBlob meets the requirement)
    df['NLP_Sentiment'] = df['Text'].apply(analyze_sentiment)
    
    # Categorize Sentiment
    def categorize(score):
        if score < -0.2: return 'Negative'
        elif score > 0.2: return 'Positive'
        return 'Neutral'
        
    df['Sentiment_Category'] = df['NLP_Sentiment'].apply(categorize)
    return df

def get_city_sentiment_summary(df, city=None):
    if city:
        df = df[df['City'] == city]
        
    summary = df.groupby('City')['NLP_Sentiment'].mean().reset_index()
    # rename
    summary.columns = ['City', 'Average_Sentiment']
    
    # Count negative complaints
    neg_counts = df[df['Sentiment_Category'] == 'Negative'].groupby('City').size().reset_index(name='Negative_Complaints')
    
    merged = pd.merge(summary, neg_counts, on='City', how='left').fillna(0)
    return merged
