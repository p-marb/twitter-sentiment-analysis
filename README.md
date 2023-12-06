# Financial/Stock Market Twitter Sentiment Analysis

This is a very basic twitter sentiment analysis that can be ran in two modes, train or run. 

The model was trained off the 'Twitter Financial News Sentiment' dataset available here: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

The model requirements can be installed running ``pip install -r requirements.txt``

# Model Outputs (Running the model)

Once trained, the model can attempt to classify given an input text sequence. Here are some different outputs:


- Text to analyze: JUST IN : UBER TO JOIN THE S&P 500
    - Sentiment: Neutral
    - Bearish: 0.42%
    - Bullish: 26.50%
    - Neutral: 73.08%
- Text to analyze: S&P 500 CLOSED AT A NEW YEARLY HIGH TODAY  SOMEONE GO CHECK ON THE PEOPLE CALLING FOR A 80% CRASH THIS YEAR
    - Sentiment: Bullish
    - Bearish: 0.40%
    - Bullish: 98.46%
    - Neutral: 1.14%

Interestingly, the model can still perform on tweets not necessarily related to the stock market to find some amount of underlying sentiment. 

- Text to analyze: NYC is so expensive that Gen Z and Millennials are taking teens’ babysitting jobs, per NYP.
    - Sentiment: Neutral
    - Bearish: 2.87%
    - Bullish: 0.10%
    - Neutral: 97.03%

As you can see, the overall sentiment is overwhelmingly Neutral, but there still is a slight Bearish sentiment the model picks up on still.