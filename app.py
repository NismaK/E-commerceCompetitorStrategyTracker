import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

API_KEY = "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"  # Groq API Key

def truncate_text(text, max_length=512):
    return text[:max_length]


def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_price_data.csv")
    print(data.head())
    return data


def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews_data.csv")
    return reviews


def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)


def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["Price"] = data["Price"].astype(int).round(2)
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

    X = data[["Price", "Discount"]]
    y = data["Predicted_Discount"]
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


import numpy as np
import pandas as pd


def forecast_discounts_arima(data, future_days=7):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    if data.empty:
        st.warning("No valid discount data available for forecasting.")
        return pd.DataFrame()  # Return an empty DataFrame

    discount_series = data["Discount"]

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # Check if discount_series is empty after processing
    if discount_series.empty:
        st.warning("No valid historical discount data for ARIMA model.")
        return pd.DataFrame()

    model = ARIMA(discount_series, order=(0, 1, 2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=pd.Timestamp.today().normalize(),  # Start from today
        periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast.round(2)})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df


def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

1. **Product Name**: {product_name}

2. **Competitor Data** (including current prices, discounts, and predicted discounts):
{competitor_data}

3. **Sentiment Analysis**:
{sentiment}


5. **Today's Date**: {str(date)}

### Task:
- Analyze the competitor data and identify key pricing trends.
- Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
- Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
- Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
- Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and outperforming competitors.

Provide your recommendations in a structured format:
1. **💰Pricing Strategy**
2. **🎯Promotional Campaign Ideas**
3. **⭐Customer Satisfaction Recommendations**
    """

    messages = [{"role": "user", "content": prompt}]

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
        timeout=10,
    )
    res = res.json()
    response = res["choices"][0]["message"]["content"]
    return response


def generate_price_recommendation(selected_product, product_data_with_predictions, sentiments):
    """Generate an optimal selling price recommendation using LLM with reduced input size."""
    
    # Reduce competitor data to the last 5 rows
    competitor_summary = product_data_with_predictions.tail(5).to_string()

    # Summarize discount trends
    avg_predicted_discount = product_data_with_predictions["Predicted_Discount"].mean()
    discount_summary = f"Average Predicted Discount: {avg_predicted_discount:.2f}% over the next week."

    # Summarize sentiment analysis
    sentiment_summary = sentiments if sentiments else "No sentiment data available."

    prompt = f"""
    You are a pricing strategist for an e-commerce platform. Your goal is to determine the best selling price for **{selected_product}**.

    **Key Insights:**
    1. **Competitor Pricing & Discounts (Last 5 Entries):**  
    {competitor_summary}

    2. **Discount Trend Summary:**  
    {discount_summary}

    3. **Customer Sentiment Analysis:**  
    {sentiment_summary}

    **Task:**
    - Identify market pricing trends.
    - Analyze how predicted discounts impact sales.
    - Adjust pricing based on customer sentiment.
    - Consider profit from sellers point of view
    - Recommend an **optimal selling price in Indian Rupees (₹)** that balances **profitability,sales and customer satisfaction**.
    
    ### Output Format:
    📌 **Optimal Selling Price**: ₹ <value>  
    💡 **Reasoning**:  
    - **Competitor Pricing**: ₹X with avg discount of Y%.  
    - **Discount Trend**: Expected Z% drop, so adjusting accordingly.  
    - **Customer Sentiment**: Users prefer products priced around ₹<range>.  
    - **Final Suggestion**: ₹<value> ensures profitability while increasing sales volume.  
    """

    chat_data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0.5,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(chat_data),
            headers=headers,
            timeout=10,
        ).json()

        return res.get("choices", [{}])[0].get("message", {}).get("content", "No price recommendation available.")
    
    except Exception as e:
        return f"Error in price recommendation: {e}"

def chatbot_response(user_query, selected_product, product_data_with_predictions, sentiments):
    """Generate chatbot response for user queries."""
    chat_prompt = f"""
You are an expert e-commerce analyst. Based on the following product data(having indian rupees as currency), answer the user's query accordingly. 
    1. **Product Name**: {selected_product}
    2. **Competitor Data**: 
    {product_data_with_predictions.to_string()}
    3. **Discount Predictions**: 
    {product_data_with_predictions['Predicted_Discount'].to_list()}
    4. **Sentiment Analysis**: 
    {sentiments if sentiments else "No sentiment data available."}
    
    ### User Question:
    {user_query}
    """
    
    chat_data = {
        "messages": [{"role": "user", "content": chat_prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0.5,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    try:
        chat_res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(chat_data),
            headers=headers,
            timeout=10,
        ).json()
        
        return chat_res.get("choices", [{}])[0].get("message", {}).get("content", "No response available.")
    
    except Exception as e:
        return f"Error in chatbot: {e}"


####-----------------------Frontend---------------------------##########

st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout= "wide")


st.title("🚀 E-Commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")


competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

products = competitor_data['Product name'].unique()

selected_product = st.sidebar.selectbox("🔍Choose a product to analyze:", products)

product_data = competitor_data[competitor_data["Product name"] == selected_product]
product_reviews = reviews_data[reviews_data["Product name"] == selected_product]

st.markdown(f"<h2 style='color: black;'>🛒 Competitor Analysis for {selected_product}</h2>", unsafe_allow_html=True)
st.subheader("💲Competitor Price Data")
st.table(product_data.tail(5).round(2).set_index(product_data.columns[0]))


# Add Expander for Price & Discount Trends
product_data["Date"] = pd.to_datetime(product_data["Date"])

# Get the latest date in the dataset
latest_date = product_data["Date"].max()
two_months_ago = latest_date - pd.DateOffset(months=2)

# Filter data for the last two months
filtered_product_data = product_data[product_data["Date"] >= two_months_ago]

with st.expander("🔍 View Competitor Price and Discount Trends"):
    st.subheader("📉 Price & Discount Trends for Flipkart & Amazon")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Price Trend")
        fig_price = px.line(
            filtered_product_data, 
            x="Date",
            y="Price", 
            color="Source",  
            title="Price Trends - Flipkart vs Amazon",
            markers=True, 
            line_shape="linear",
            color_discrete_map={"Flipkart": "blue", "Amazon": "orange"}
        )
        fig_price.update_xaxes(
            tickformat="%Y-%m-%d",
            tickangle=45  
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.markdown("### 📊 Discount Trend")
        fig_discount = px.line(
            filtered_product_data, 
            x="Date",
            y="Discount", 
            color="Source",  
            title="Discount Trends - Flipkart vs Amazon",
            markers=True, 
            line_shape="linear",
            color_discrete_map={"Flipkart": "blue", "Amazon": "orange"}
        )
        fig_discount.update_xaxes(
            tickformat="%Y-%m-%d",
            tickangle=45  
        )
        st.plotly_chart(fig_discount, use_container_width=True)
       
if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)
    )
    reviews = product_reviews["reviews"].tolist()
    sentiments = analyze_sentiment(reviews)

    st.subheader("😊 Customer Sentiment Analysis")
    
    # Convert sentiment results into a DataFrame
    sentiment_df = pd.DataFrame(sentiments)
    
    # Define colors for sentiment categories
    color_map = {"POSITIVE": "green", "NEGATIVE": "red"}  
    
    # Create bar chart with custom colors
    fig = px.bar(
        sentiment_df, 
        x="label", 
        y="score",  # Ensure y-axis represents the sentiment score
        title="Sentiment Analysis Results",
        color="label",
        color_discrete_map=color_map
    )
    
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")

# Preprocessing

product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
product_data = product_data.dropna(subset=["Date"])
product_data.set_index("Date", inplace=True)
product_data = product_data.sort_index()

product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model
product_data_with_predictions = forecast_discounts_arima(product_data)


st.subheader("🏷️Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.style.format({"Predicted_Discount": "{:.2f}"}))
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)

st.subheader("📢 Strategic Recommendations")
st.write(recommendations)

st.subheader("💰 Optimal Price Recommendation")

# Add a button to generate the price recommendation
if st.button("🔍 Get Optimal Price"):
    price_recommendation = generate_price_recommendation(
        selected_product, 
        product_data_with_predictions, 
        sentiments if not product_reviews.empty else "No reviews available"
    )

    st.write("### 📢 Recommended Selling Price")
    st.write(price_recommendation)

# Sidebar: Enable Chatbot Toggle
enable_chatbot = st.sidebar.checkbox("💬 Enable Chatbot", value=False, key="enable_chatbot")

# If chatbot is enabled, show the input box outside the sidebar
if enable_chatbot:
    st.subheader("💬 Chat with AI")
    user_query = st.text_area("Ask about pricing and discounts:", key="chat_input")
    submit_query = st.button("Submit", key="submit_chat")

    if submit_query and user_query:
        response = chatbot_response(user_query, selected_product, product_data_with_predictions, sentiments)
        st.write("### 🤖 Chatbot Response:")
        st.write(response)
