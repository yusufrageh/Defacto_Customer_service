import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.defacto.com.eg"
CATEGORY_URL = "https://www.defacto.com.eg/man"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Step 1: Extract product links from main category page
def get_product_links():
    response = requests.get(CATEGORY_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    product_cards = soup.select("div.product-card a.card-image")  # Link containers
    links = [BASE_URL + a["href"] for a in product_cards if a.get("href")]
    return list(set(links))  # remove duplicates

# Step 2: Extract data from each product page
def extract_product_data(link):
    response = requests.get(link, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    
    try:
        name = soup.select_one("h1.product-name").get_text(strip=True)
    except:
        name = ""

    try:
        code = soup.select_one("div.product-code").get_text(strip=True).replace("Product Code: ", "")
    except:
        code = link.split("/")[-1]  # fallback

    try:
        image_url = soup.select_one("div.carousel-item img")["src"]
    except:
        image_url = ""

    try:
        sizes = [size.get_text(strip=True) for size in soup.select("div.size-list span.label")]
        sizes_str = ", ".join(sizes)
    except:
        sizes_str = ""

    return {
        "product_name": name,
        "product_code": code,
        "image_url": image_url,
        "product_link": link,
        "available_sizes": sizes_str
    }

# Step 3: Loop over all products and collect data
def run_etl():
    print("Extracting product links...")
    links = get_product_links()
    print(f"Found {len(links)} products.")

    all_data = []
    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] Scraping: {link}")
        try:
            data = extract_product_data(link)
            all_data.append(data)
            time.sleep(1)  # be polite to the server
        except Exception as e:
            print(f"Error with {link}: {e}")

    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv("men_clothes.csv", index=False, encoding="utf-8")
    print("Saved to men_clothes.csv")

if __name__ == "__main__":
    run_etl()
