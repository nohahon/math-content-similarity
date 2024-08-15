import csv
import requests

def query_zbmath(query, api_key):
    """ Query the zbMATHOpen API and return the JSON response. """
    url = f"https://zbmath.org/api/v1/search?q={query}&format=json"
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data for {query}: {response.status_code}")
        return None

def main(input_csv_path, output_csv_path, api_key):
    results = []
    
    with open(input_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:  # Skip empty rows
                continue
            query = row[0]
            result = query_zbmath(query, api_key)
            if result:
                # Example of extracting some information from the result
                articles = result.get('articles', [])
                for article in articles:
                    results.append([query, article.get('title', ''), article.get('authors', '')])

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Query', 'Title', 'Authors'])  # Column headers
        writer.writerows(results)

# Please request an API key from zbMATH Open
if __name__ == "__main__":
    input_csv_file = 'input.csv'
    output_csv_file = 'output.csv'
    api_key = 'YOUR_API_KEY_HERE'
    main(input_csv_file, output_csv_file, api_key)