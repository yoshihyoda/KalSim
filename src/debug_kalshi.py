
import requests
import json

KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

def fetch_debug_data():
    try:
        response = requests.get(KALSHI_API_URL, params={"limit": 5})
        response.raise_for_status()
        data = response.json()
        markets = data.get("markets", [])
        
        for i, market in enumerate(markets[:5]):
            print(f"--- Market {i+1} ---")
            print("Ticker:", market.get("ticker"))
            print("Event Ticker:", market.get("event_ticker"))
            print("MVE:", market.get("mve_collection_ticker"))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_debug_data()
