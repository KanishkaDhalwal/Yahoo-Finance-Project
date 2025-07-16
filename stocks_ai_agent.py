
import yfinance as yf
import pandas as pd
import openai
import os
from openai import OpenAI

# ----------- Configuration -----------
openai.api_key =("OPENAI_API_KEY")  
MODEL = "gpt-4.1"
NUM_ROWS = 30  # Number of rows to include in the LLM prompt
# -------------------------------------


def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def summarize_data_for_prompt(df: pd.DataFrame, limit=NUM_ROWS) -> str:
    df_sample = df.tail(limit)
    return df_sample.to_string(index=False)


def generate_prompt(ticker, start_date, end_date, data_snippet):
    return f"""
You are an expert financial AI model. Given the following OHLC stock data from {start_date} to {end_date} for {ticker}:

{data_snippet}

Step-by-step:
1. Analyze trends and patterns in closing prices.
2. Identify significant support/resistance levels.
3. Determine if the stock is currently overbought, oversold, or in consolidation.
4. Predict the short-term direction with reasoning.
5. Give a confidence level (0-100%) for the prediction.

Respond in structured format with bullet points and a summary paragraph at the end.
"""


client = OpenAI(api_key=("OPENAI_API_KEY"))


def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a seasoned financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g. MSFT): ").upper()
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    print("\nFetching stock data...")
    df = get_stock_data(ticker, start_date, end_date)
    data_snippet = summarize_data_for_prompt(df)

    print("\nGenerating prompt and sending to LLM...")
    prompt = generate_prompt(ticker, start_date, end_date, data_snippet)
    response = ask_llm(prompt)

    print("\nLLM Analysis and Forecast:\n")
    print(response)
