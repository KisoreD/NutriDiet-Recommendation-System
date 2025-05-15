from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Data Sets/Final Data set.csv")
Nutrient_recommend = pd.read_csv("Data Sets/Micro and macro nutrients.csv")
Meal_suggestions = pd.read_csv("Data Sets/Meal suggestions.csv")
RandomForest = pickle.load(open("model/RandomForest1.pkl", 'rb'))
app = Flask(__name__)

VALID_OPTIONS = {
    'Gender': {
        'Male': 0,
        'Female': 1
    },
    'Diet_Preference': {
        'Non-Vegetarian': 0,
        'Vegetarian': 1,
        'Vegan': 2,
        'Plant-Based': 3,
        'Mediterranean': 4,
        'Keto': 5,
        'Paleo': 6,
        'Low-Carb': 7,
        'Gluten-Free': 8,
        'Pescatarian': 9,
        'Flexitarian': 10,
        'High-Protein': 11,
        'DASH': 12,
        'South Asian': 13,
        'East Asian': 14,
        'Raw Vegan': 15,
        'Halal': 16,
        'Kosher': 17,
        'Low-FODMAP': 18,
        'Mediterranean-Vegetarian': 19
    },
    'Activity_Level': {
        'Sedentary': 0,
        'Light': 1,
        'Moderate': 2,
        'Active': 3,
        'Very Active': 4
    },
    'Disease': {
        'None': 0,
        'Diabetes': 1,
        'Hypertension': 2,
        'Heart Disease': 3,
        'GERD': 4,
        'IBS': 5,
        'Celiac': 6,
        'Obesity': 7,
        'Thyroid': 8,
        'Iron Deficiency': 9,
        'B12 Deficiency': 10,
        'Food Allergies': 11
    },
    'Food_Allergies': {
        'None': 0,
        'Peanuts': 1,
        'Tree Nuts': 2,
        'Milk': 3,
        'Eggs': 4,
        'Soy': 5,
        'Fish': 6,
        'Shellfish': 7,
        'Wheat': 8,
        'Multiple': 9
    },
    'Health_Goal': {
        'Weight Loss': 0,
        'Maintenance': 1,
        'Weight Gain': 2,
        'Muscle Gain': 3,
        'Better Health': 4,
        'Athletic Performance': 5
    }
}

MEAL_SUGGESTIONS = {
    'Vegetarian': {
        'breakfast': [
            'Oatmeal with protein powder, banana, nuts and seeds',
            'Greek yogurt parfait with granola, berries and honey',
            'Whole grain toast with avocado, scrambled eggs and spinach',
            'Protein smoothie bowl with mixed fruits, granola and chia seeds',
            'Cottage cheese pancakes with fresh fruits and maple syrup'
        ],
        'lunch': [
            'Quinoa bowl with roasted chickpeas, mixed vegetables and tahini dressing',
            'Lentil curry with brown rice and steamed vegetables',
            'Greek salad with feta cheese, olives and whole grain pita',
            'Vegetable stir-fry with tofu and brown rice',
            'Black bean and sweet potato burrito bowl with guacamole'
        ],
        'dinner': [
            'Tofu stir-fry with brown rice and mixed vegetables',
            'Chickpea curry with quinoa and roasted vegetables',
            'Black bean burgers with sweet potato wedges and salad',
            'Spinach and ricotta stuffed pasta with tomato sauce',
            'Buddha bowl with tempeh, roasted vegetables and tahini dressing'
        ],
        'snacks': [
            'Greek yogurt with honey and mixed nuts',
            'Protein smoothie with banana and peanut butter',
            'Trail mix with nuts, seeds and dried fruits',
            'Cottage cheese with fruit and granola',
            'Hummus with carrot sticks and whole grain crackers'
        ]
    },
    'Vegan': {
        'breakfast': [
            'Tofu scramble with vegetables and whole grain toast',
            'Overnight oats with plant milk, fruits and chia seeds',
            'Protein smoothie bowl with plant milk and granola',
            'Quinoa porridge with berries and almond butter',
            'Whole grain toast with avocado and tempeh bacon'
        ],
        'lunch': [
            'Buddha bowl with quinoa, tempeh and vegetables',
            'Lentil and vegetable curry with brown rice',
            'Chickpea salad sandwich with avocado',
            'Tofu and vegetable stir-fry with noodles',
            'Black bean and sweet potato tacos'
        ],
        'dinner': [
            'Tempeh stir-fry with brown rice and vegetables',
            'Lentil shepherd\'s pie with mushrooms',
            'Chickpea curry with quinoa',
            'Black bean burgers with sweet potato fries',
            'Pasta with vegetable and cashew cream sauce'
        ],
        'snacks': [
            'Mixed nuts and dried fruits',
            'Plant protein smoothie',
            'Hummus with vegetable sticks',
            'Energy balls made with dates and nuts',
            'Roasted chickpeas'
        ]
    }
}


def helper(calories):
    nutrient_info = Nutrient_recommend.iloc[(Nutrient_recommend['Daily_Calories'] - calories).abs().argsort()[:1]].iloc[
        0]
    meal_info = Meal_suggestions.iloc[(Meal_suggestions['Daily_Calories'] - calories).abs().argsort()[:1]].iloc[0]


    formatted_nutrients = {
        'Calories': f"<p><strong>Daily Calories:</strong> {nutrient_info['Daily_Calories']} kcal</p>",
        'Protein': f"<p><strong>Protein:</strong> {nutrient_info['Protein_g']} g</p>",
        'Carbs': f"<p><strong>Carbohydrates:</strong> {nutrient_info['Carbs_g']} g</p>",
        'Fat': f"<p><strong>Fat:</strong> {nutrient_info['Fat_g']} g</p>",
        'Fiber': f"<p><strong>Fiber:</strong> {nutrient_info['Fiber_g']} g</p>",
        'Sugar': f"<p><strong>Sugar:</strong> {nutrient_info['Sugar_g']} g</p>",
        'VitaminA': f"<p><strong>Vitamin A:</strong> {nutrient_info['Vitamin_A_mcg']} mcg</p>",
        'VitaminC': f"<p><strong>Vitamin C:</strong> {nutrient_info['Vitamin_C_mg']} mg</p>",
        'VitaminD': f"<p><strong>Vitamin D:</strong> {nutrient_info['Vitamin_D_mcg']} mcg</p>",
        'Calcium': f"<p><strong>Calcium:</strong> {nutrient_info['Calcium_mg']} mg</p>",
        'Iron': f"<p><strong>Iron:</strong> {nutrient_info['Iron_mg']} mg</p>",
        'Potassium': f"<p><strong>Potassium:</strong> {nutrient_info['Potassium_mg']} mg</p>",
        'Magnesium': f"<p><strong>Magnesium:</strong> {nutrient_info['Magnesium_mg']} mg</p>",
        'Zinc': f"<p><strong>Zinc:</strong> {nutrient_info['Zinc_mg']} mg</p>"
    }
    return formatted_nutrients, meal_info


def get_meal_suggestions(diet_pref, calories):
    diet_category = diet_pref
    if diet_category not in MEAL_SUGGESTIONS:
        diet_category = 'Vegetarian'
    meals = MEAL_SUGGESTIONS[diet_category]

    formatted_meals = {
        'Breakfast': f"<p><strong>Breakfast:</strong> {random.choice(meals['breakfast'])}</p>",
        'Lunch': f"<p><strong>Lunch:</strong> {random.choice(meals['lunch'])}</p>",
        'Dinner': f"<p><strong>Dinner:</strong> {random.choice(meals['dinner'])}</p>",
        'Snacks': f"<p><strong>Snacks:</strong> {random.choice(meals['snacks'])}</p>"
    }
    return formatted_meals


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            age = request.form.get('Age')
            gender = request.form.get('Gender')
            weight = request.form.get('Weight')
            height = request.form.get('Height')
            diet_preference = request.form.get('Diet_Preference')
            activity_level = request.form.get('Activity_Level')
            weekly_activity = request.form.get('Weekly_Activity_Days')
            disease = request.form.get('Disease')
            allergies = request.form.get('Food_Allergies')
            health_goal = request.form.get('Health_Goal')

            user_data = {
                'Age': int(age),
                'Gender': gender,
                'Weight_kg': float(weight),
                'Height_cm': float(height),
                'Diet_Preference': diet_preference,
                'Activity_Level': activity_level,
                'Weekly_Activity_Days': int(weekly_activity),
                'Disease': disease,
                'Food_Allergies': allergies,
                'Health_Goal': health_goal
            }

            input_df = pd.DataFrame([user_data])

            categorical_columns = ['Gender', 'Diet_Preference', 'Activity_Level', 'Disease', 'Food_Allergies',
                                   'Health_Goal']
            for column in categorical_columns:
                input_df[column] = VALID_OPTIONS[column][input_df[column].iloc[0]]

            calories = int(RandomForest.predict(input_df)[0])
            nutrient_info, meal_info = helper(calories)
            meal_suggestions = get_meal_suggestions(diet_preference, calories)

            return render_template('index.html',
                                   calories=calories,
                                   nutrient_info=nutrient_info,
                                   meal_suggestions=meal_suggestions)

        except Exception as e:
            return f"An error occurred: {str(e)}"
    return render_template('index.html')





if __name__ == "__main__":
    app.run(debug=True)