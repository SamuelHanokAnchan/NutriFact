import requests

API_KEY = "lEQzDPuI9WbNZ4Iis6C6YNcL2gWWyc4cLjmuAfmF"
SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

def get_nutrition(food_name):
    response = requests.get(
        SEARCH_URL,
        params={
            "query": food_name,
            "pageSize": 1,
            "api_key": API_KEY
        }
    )

    data = response.json()
    foods = data.get("foods", [])

    if not foods:
        print("No data found.")
        return

    nutrients = foods[0]["foodNutrients"]

    def get_value(name):
        for n in nutrients:
            if name.lower() in n["nutrientName"].lower():
                return n.get("value", 0)
        return 0

    print(f"\nNutrition for: {food_name.title()}")
    print("-" * 30)
    print(f"Calories : {get_value('Energy')} kcal")
    print(f"Protein  : {get_value('Protein')} g")
    print(f"Carbs    : {get_value('Carbohydrate')} g")
    print(f"Fat      : {get_value('Total lipid')} g")
    print(f"Fiber    : {get_value('Fiber')} g")
    print(f"Sugar    : {get_value('Sugars')} g")

if __name__ == "__main__":
    food = input("Enter food name: ")
    get_nutrition(food)
