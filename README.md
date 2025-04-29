# gold_predictor
Kivy app code for Gold Predictor Pro
# -------------------
# 🚀 Gold Price Predictor App - Iran Market
# Full Python Code - By ChatGPT
# -------------------

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import schedule
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# -------------------
# Step 1: Scrape Gold Prices
# -------------------

def get_gold_price():
    url = "https://www.tgju.org/profile/geram18"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    try:
        price_element = soup.find("span", {"id": "last_current"})
        price_text = price_element.text.replace(",", "")
        price = float(price_text)
        return price
    except Exception as e:
        print("Error scraping gold price:", e)
        return None

# -------------------
# Step 2: Scrape News Headlines
# -------------------

def get_news_sentiment():
    urls = [
        "https://www.reuters.com/",
        "https://www.bloomberg.com/",
        "https://edition.cnn.com/world",
        "https://www.bbc.com/news/world",
        "https://www.aljazeera.com/news/",
        "https://www.ft.com/world"
    ]
    sia = SentimentIntensityAnalyzer()
    sentiments = []

    for url in urls:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.content, "html.parser")
            headlines = soup.find_all("h3")
            for headline in headlines:
                text = headline.get_text()
                if any(word in text.lower() for word in ["iran", "us", "nuclear", "deal", "negotiation"]):
                    score = sia.polarity_scores(text)['compound']
                    sentiments.append(score)
        except:
            continue

    if sentiments:
        avg_sentiment = np.mean(sentiments)
    else:
        avg_sentiment = 0
    
    return avg_sentiment

# -------------------
# Step 3: Predict Future Prices
# -------------------

def predict_prices(price_list, news_sentiment):
    if len(price_list) < 3:
        return None, None
    
    X = np.array(range(len(price_list))).reshape(-1, 1)
    y = np.array(price_list)

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 1 day and 3 days
    next_day = model.predict([[len(price_list) + 1]])[0]
    three_days = model.predict([[len(price_list) + 3]])[0]

    # Adjust prediction based on news sentiment
    adjustment = (1 + news_sentiment * 0.05)  # 5% impact max
    next_day_adjusted = next_day * adjustment
    three_days_adjusted = three_days * adjustment

    return next_day_adjusted, three_days_adjusted

# -------------------
# Step 4: Suggest Action
# -------------------

def suggest_action(current_price, predicted_price):
    change_percent = (predicted_price - current_price) / current_price * 100
    if change_percent > 1:
        return "Buy (Expected Increase)"
    elif change_percent < -1:
        return "Sell (Expected Decrease)"
    else:
        return "Hold (Stable Prediction)"

# -------------------
# Step 5: Main Function
# -------------------

def run_app():
    gold_prices = []

    # Initial data collection
    for _ in range(5):
        price = get_gold_price()
        if price:
            gold_prices.append(price)
        time.sleep(2)

    news_sentiment = get_news_sentiment()

    next_day, three_days = predict_prices(gold_prices, news_sentiment)
    
    if next_day and three_days:
        current_price = gold_prices[-1]
        print(f"Current Price: {current_price} IRR per gram")
        print(f"Predicted Price for Tomorrow: {next_day:.2f} IRR")
        print(f"Predicted Price for 3 Days Later: {three_days:.2f} IRR")

        action = suggest_action(current_price, next_day)
        print(f"Suggested Action: {action}")

        # Plot prices
        days = list(range(len(gold_prices))) + [len(gold_prices) + 1, len(gold_prices) + 3]
        prices = gold_prices + [next_day, three_days]

        plt.figure(figsize=(10,5))
        plt.plot(days, prices, marker='o')
        plt.title("Gold Price Forecast (IRR per gram)")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()
    else:
        print("Not enough data to predict.")

# -------------------
# Run the Application
# -------------------

if __name__ == "__main__":
    run_app()
Initial commit – upload Kivy app code
