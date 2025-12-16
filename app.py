"""
FoodVisionAI - Streamlit Application
Automated Nutritional Analysis with AI-powered Food Recognition
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import base64
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="FoodVisionAI", page_icon="🥗", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = "models/food_classifier.keras"
CLASSES_PATH = "models/food_classes.json"
IMAGE_SIZE = 224

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
RECOMMENDED_WATER_CUPS = 8

# Default serving size is 100g (API returns per 100g)
DEFAULT_SERVING_GRAMS = 100

ICONS = {
    'flame': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>',
    'droplet': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>',
    'heart': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>',
    'camera': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>',
    'upload': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
    'user': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    'target': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    'activity': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    'calendar': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>',
    'utensils': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/></svg>',
    'sparkles': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>',
    'leaf': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    'scale': '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
}

def icon(name, size=24, color="currentColor"):
    svg = ICONS.get(name, ICONS['heart'])
    svg = svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')
    if color != "currentColor":
        svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    return svg

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main-header { font-size: 2.75rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.25rem; }
    .sub-header { font-size: 1rem; color: #64748b; text-align: center; margin-bottom: 2.5rem; }
    .metric-card { background: linear-gradient(145deg, #fff, #f8fafc); border: 1px solid #e2e8f0; border-radius: 20px; padding: 1.75rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem; }
    .metric-value { font-size: 2.25rem; font-weight: 700; color: #0f172a; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
    .food-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.25rem; margin-bottom: 0.75rem; border-left: 4px solid #6366f1; }
    .meal-tag { display: inline-flex; padding: 0.375rem 0.875rem; border-radius: 100px; font-size: 0.75rem; font-weight: 600; margin-right: 0.75rem; text-transform: uppercase; }
    .breakfast-tag { background: #fef3c7; color: #92400e; }
    .lunch-tag { background: #d1fae5; color: #065f46; }
    .dinner-tag { background: #dbeafe; color: #1e40af; }
    .snack-tag { background: #fce7f3; color: #9d174d; }
    .progress-container { background: #f1f5f9; border-radius: 100px; height: 12px; overflow: hidden; margin: 0.5rem 0; }
    .progress-bar { height: 100%; border-radius: 100px; transition: width 0.5s ease; }
    .progress-calories { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
    .progress-water { background: linear-gradient(90deg, #0ea5e9, #06b6d4); }
    .progress-protein { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .progress-carbs { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .progress-fat { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .step-indicator { display: flex; justify-content: center; gap: 0.75rem; margin-bottom: 2rem; }
    .step-dot { width: 10px; height: 10px; border-radius: 100px; background: #e2e8f0; }
    .step-dot.active { width: 32px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }
    .stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border: none; border-radius: 12px; padding: 0.875rem 2rem; font-weight: 600; width: 100%; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.35); }
    .prediction-result { background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7); color: white; border-radius: 20px; padding: 2rem; text-align: center; margin: 1.5rem 0; }
    .prediction-food { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; }
    .prediction-confidence { font-size: 0.875rem; opacity: 0.9; }
    .section-title { font-size: 1.125rem; font-weight: 700; color: #0f172a; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
    .empty-state { text-align: center; padding: 3rem 2rem; color: #94a3b8; background: #f8fafc; border-radius: 20px; border: 2px dashed #e2e8f0; }
    .icon-box { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }
    .icon-box.purple { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; }
    .icon-box.blue { background: linear-gradient(135deg, #0ea5e9, #06b6d4); color: white; }
    .icon-box.green { background: linear-gradient(135deg, #10b981, #34d399); color: white; }
    .icon-box.orange { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: white; }
    .icon-box.pink { background: linear-gradient(135deg, #ec4899, #f472b6); color: white; }
    .divider { height: 1px; background: #e2e8f0; margin: 2rem 0; }
    .ai-advice-card { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 1px solid #a7f3d0; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; }
    .ai-advice-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; font-weight: 700; color: #065f46; }
    .ai-advice-content { color: #047857; font-size: 0.9rem; line-height: 1.6; }
    .ai-advice-unhealthy { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 1px solid #fecaca; }
    .ai-advice-unhealthy .ai-advice-header { color: #991b1b; }
    .ai-advice-unhealthy .ai-advice-content { color: #b91c1c; }
    .ai-advice-moderate { background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1px solid #fde68a; }
    .ai-advice-moderate .ai-advice-header { color: #92400e; }
    .ai-advice-moderate .ai-advice-content { color: #b45309; }
    .alternative-food { background: #fff; border-radius: 12px; padding: 1rem; margin-top: 1rem; border: 1px solid #e2e8f0; }
    .alternative-title { font-weight: 600; color: #0f172a; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
    .quantity-box { background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border: 1px solid #bae6fd; border-radius: 16px; padding: 1.25rem; margin: 1rem 0; }
    .quantity-label { font-size: 0.875rem; font-weight: 600; color: #0369a1; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
    .quantity-value { font-size: 1.5rem; font-weight: 700; color: #0c4a6e; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    defaults = {
        'onboarding_step': 1, 'onboarding_complete': False, 'user_profile': {'health_conditions': []},
        'daily_calories_consumed': 0.0, 'daily_protein': 0.0, 'daily_carbs': 0.0, 'daily_fat': 0.0,
        'water_cups': 0, 'meal_history': [], 'last_reset_date': str(date.today()),
        'prediction_result': None, 'nutrition_data': None, 'use_fallback': False, 'ai_health_advice': None,
        'food_quantity_grams': 100  # Default quantity in grams
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.last_reset_date != str(date.today()):
        reset_daily_data()

def reset_daily_data():
    st.session_state.daily_calories_consumed = 0.0
    st.session_state.daily_protein = 0.0
    st.session_state.daily_carbs = 0.0
    st.session_state.daily_fat = 0.0
    st.session_state.water_cups = 0
    st.session_state.meal_history = []
    st.session_state.last_reset_date = str(date.today())

def calculate_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    multipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'very_active': 1.725}
    return bmr * multipliers.get(activity_level, 1.2)

def calculate_daily_calorie_target(profile):
    bmr = calculate_bmr(profile['weight'], profile['height'], profile['age'], profile.get('gender', 'male'))
    tdee = calculate_tdee(bmr, profile['activity_level'])
    if profile['goal'] == 'lose':
        weeks = profile.get('timeline_weeks', 12)
        weight_to_lose = profile['weight'] - profile.get('target_weight', profile['weight'])
        daily_deficit = min((weight_to_lose * 7700) / weeks / 7, 750)
        return int(tdee - daily_deficit)
    elif profile['goal'] == 'gain':
        return int(tdee + 500)
    return int(tdee)

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"

def get_scaled_nutrition(nutrition_data, quantity_grams):
    """Scale nutrition values based on quantity (API returns per 100g)."""
    if not nutrition_data:
        return None
    multiplier = quantity_grams / DEFAULT_SERVING_GRAMS
    return {
        "calories": safe_float(nutrition_data.get('calories', 0)) * multiplier,
        "protein_g": safe_float(nutrition_data.get('protein_g', 0)) * multiplier,
        "carbohydrates_total_g": safe_float(nutrition_data.get('carbohydrates_total_g', 0)) * multiplier,
        "fat_total_g": safe_float(nutrition_data.get('fat_total_g', 0)) * multiplier,
        "fiber_g": safe_float(nutrition_data.get('fiber_g', 0)) * multiplier,
        "sugar_g": safe_float(nutrition_data.get('sugar_g', 0)) * multiplier
    }

def get_ai_health_advice(food_name, nutrition_data, user_profile, quantity_grams=100):
    """Get personalized health advice from AI using OpenRouter Gemini."""
    try:
        scaled_nutrition = get_scaled_nutrition(nutrition_data, quantity_grams)

        bmi = calculate_bmi(
            user_profile.get('weight', 70),
            user_profile.get('height', 170)
        )
        bmi_category = get_bmi_category(bmi)

        goal_text = {
            'lose': 'lose weight',
            'gain': 'gain weight/muscle',
            'maintain': 'maintain current weight'
        }.get(user_profile.get('goal', 'maintain'), 'maintain weight')

        health_conditions = user_profile.get('health_conditions', [])
        health_text = ", ".join(health_conditions) if health_conditions else "None"

        prompt = f"""
            You are a nutritionist AI.

            Analyze the food and give personalized advice.
            Consider the user's health conditions.

            USER:
            Name: {user_profile.get('name', 'User')}
            BMI: {bmi:.1f} ({bmi_category})
            Goal: {goal_text}
            Health Conditions: {health_text}

            FOOD:
            {food_name.replace('_', ' ').title()} ({quantity_grams}g)

            Calories: {scaled_nutrition.get('calories', 0):.0f}
            Protein: {scaled_nutrition.get('protein_g', 0):.1f}g
            Carbs: {scaled_nutrition.get('carbohydrates_total_g', 0):.1f}g
            Sugar: {scaled_nutrition.get('sugar_g', 0):.1f}g
            Fat: {scaled_nutrition.get('fat_total_g', 0):.1f}g

            Reply ONLY in JSON:
            {{
            "health_rating": "healthy or moderate or unhealthy",
            "reason": "one sentence",
            "advice": "two sentences",
            "alternative": "one food or None needed"
            }}
            """

        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 400
            },
            timeout=30
        )

        result = response.json()
        ai_text = result["choices"][0]["message"]["content"]

        # ✅ FINAL CLEANING (THIS FIXES EVERYTHING)
        ai_text = ai_text.strip()
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        advice_data = json.loads(ai_text)

        return {
            "health_rating": advice_data.get("health_rating", "moderate"),
            "reason": advice_data.get("reason", ""),
            "advice": advice_data.get("advice", ""),
            "alternative": advice_data.get("alternative", "None needed")
        }

    except Exception as e:
        return {
            "health_rating": "moderate",
            "reason": "AI analysis unavailable.",
            "advice": f"Enjoy your {food_name.replace('_', ' ')} in moderation.",
            "alternative": "None needed"
        }




def render_ai_health_advice(advice_data):
    rating = advice_data.get('health_rating', 'moderate')
    if rating == 'healthy':
        card_class, rating_emoji, rating_text = 'ai-advice-card', '✅', 'Healthy Choice!'
    elif rating == 'unhealthy':
        card_class, rating_emoji, rating_text = 'ai-advice-card ai-advice-unhealthy', '⚠️', 'Consider Alternatives'
    else:
        card_class, rating_emoji, rating_text = 'ai-advice-card ai-advice-moderate', '💡', 'Moderate Choice'
    
    html = f'''<div class="{card_class}"><div class="ai-advice-header">{icon("sparkles", 20)} AI Health Advisor {rating_emoji} {rating_text}</div>
        <div class="ai-advice-content"><p style="margin:0 0 0.5rem 0;"><strong>Analysis:</strong> {advice_data.get('reason', '')}</p>
        <p style="margin:0;">{advice_data.get('advice', '')}</p></div>'''
    
    if rating != 'healthy' and advice_data.get('alternative', 'None needed') != 'None needed':
        html += f'''<div class="alternative-food"><div class="alternative-title">{icon("leaf", 18, "#10b981")} Healthier Alternative</div>
            <p style="margin:0;color:#64748b;font-size:0.875rem;">{advice_data.get('alternative', '')}</p></div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from tensorflow import keras
        if not Path(MODEL_PATH).exists(): return None, None, "Model not found"
        if not Path(CLASSES_PATH).exists(): return None, None, "Classes not found"
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f: class_names = json.load(f)
        return model, class_names, None
    except Exception as e: return None, None, str(e)

def preprocess_image(image):
    img = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def predict_food_local(model, class_names, image):
    try:
        predictions = model.predict(preprocess_image(image), verbose=0)[0]
        top_idx = np.argmax(predictions)
        return class_names[top_idx], float(predictions[top_idx]), None
    except Exception as e: return None, 0, str(e)

def predict_food_fallback(image):
    try:
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        response = requests.post(OPENROUTER_API_URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": "google/gemini-2.0-flash-001", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What food is this? Reply with ONLY the dish name, nothing else."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]}]}, timeout=30)
        return response.json()['choices'][0]['message']['content'].strip(), 0.85, None
    except Exception as e: return None, 0, str(e)

def get_nutrition_data(food_name):
    try:
        response = requests.get(USDA_SEARCH_URL, params={"query": food_name.replace("_", " "), "pageSize": 1, "api_key": USDA_API_KEY}, timeout=10)
        foods = response.json().get("foods", [])
        if not foods: return None
        nutrients = foods[0].get("foodNutrients", [])
        def get_value(name):
            for n in nutrients:
                if name.lower() in n.get("nutrientName", "").lower(): return n.get("value", 0)
            return 0
        return {"calories": get_value("Energy"), "protein_g": get_value("Protein"), "carbohydrates_total_g": get_value("Carbohydrate"),
                "fat_total_g": get_value("Total lipid"), "fiber_g": get_value("Fiber"), "sugar_g": get_value("Sugars")}
    except: return None

def render_step_indicator(current_step, total_steps=4):
    dots = ''.join([f'<div class="step-dot {"active" if i <= current_step else ""}"></div>' for i in range(1, total_steps + 1)])
    st.markdown(f'<div class="step-indicator">{dots}</div>', unsafe_allow_html=True)

def render_progress_bar(current, target, css_class, label):
    pct = min((current / target) * 100, 100) if target > 0 else 0
    st.markdown(f'''<div style="margin-bottom:1.25rem;"><div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
        <span style="font-size:0.8125rem;color:#64748b;">{label}</span><span style="font-size:0.8125rem;font-weight:600;color:#0f172a;">{current:.0f}/{target:.0f}</span></div>
        <div class="progress-container"><div class="progress-bar {css_class}" style="width:{pct}%;"></div></div></div>''', unsafe_allow_html=True)

def render_meal_card(meal):
    tags = {'breakfast': 'breakfast-tag', 'lunch': 'lunch-tag', 'dinner': 'dinner-tag', 'snack': 'snack-tag'}
    calories = safe_float(meal.get('calories', 0))
    quantity = meal.get('quantity_grams', 100)
    st.markdown(f'''<div class="food-card"><div style="display:flex;justify-content:space-between;align-items:center;">
        <div><span class="meal-tag {tags.get(meal['meal_type'], 'snack-tag')}">{meal['meal_type'].capitalize()}</span>
        <span style="font-weight:600;color:#0f172a;">{meal['food_name'].replace('_', ' ').title()}</span>
        <span style="font-size:0.75rem;color:#94a3b8;margin-left:0.5rem;">({quantity}g)</span></div>
        <div style="text-align:right;"><div style="font-weight:700;color:#6366f1;">{calories:.0f} cal</div>
        <div style="font-size:0.75rem;color:#94a3b8;">{meal['time']}</div></div></div></div>''', unsafe_allow_html=True)

def render_onboarding():
    st.markdown('''<div style="display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:0.5rem;">
        <div style="width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#22c55e,#16a34a);display:flex;align-items:center;justify-content:center;font-size:28px;color:white;box-shadow:0 10px 25px rgba(34,197,94,0.35);">🥗</div>
        <h1 class="main-header" style="margin:0;">FoodVisionAI</h1></div>''', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-Powered Nutrition Companion</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: render_step_indicator(st.session_state.onboarding_step)
    step = st.session_state.onboarding_step
    
    if step == 1:
        st.markdown(
            f'''
            <div style="text-align:center;margin-bottom:1.5rem;">
                <div class="icon-box purple" style="margin:0 auto 1rem;width:56px;height:56px;border-radius:16px;">
                    {icon("user", 28, "white")}
                </div>
                <h3 style="margin:0;font-size:1.25rem;font-weight:700;">
                    Tell us about yourself
                </h3>
            </div>
            ''',
            unsafe_allow_html=True
        )

        name = st.text_input("Name", placeholder="Enter your name")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 10, 120, 25)
            weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0, 0.5)

        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, 0.5)

        # 🔽 HEALTH CONDITIONS FORM (NEW)
        st.markdown("### Health Conditions (Optional)")
        health_conditions = st.multiselect(
            "Do you have any of the following?",
            [
                "Diabetes",
                "High Blood Pressure (Hypertension)",
                "Heart Disease",
                "High Cholesterol",
                "Obesity",
                "Kidney Disease",
                "Lactose Intolerance",
                "Gluten Sensitivity"
            ],
            help="This information helps the AI give you safer food advice"
        )

        if st.button("Continue", key="s1") and name:
            st.session_state.user_profile.update({
                'name': name,
                'age': age,
                'weight': weight,
                'height': height,
                'gender': gender.lower(),
                'health_conditions': health_conditions  # ✅ STORED HERE
            })
            st.session_state.onboarding_step = 2
            st.rerun()

    elif step == 2:
        st.markdown(f'<div style="text-align:center;margin-bottom:1.5rem;"><div class="icon-box green" style="margin:0 auto 1rem;width:56px;height:56px;border-radius:16px;">{icon("target", 28, "white")}</div><h3 style="margin:0;font-size:1.25rem;font-weight:700;">What\'s your goal?</h3></div>', unsafe_allow_html=True)
        goal = st.radio("Goal:", ['lose', 'gain', 'maintain'], format_func=lambda x: {'lose': 'Lose Weight', 'gain': 'Gain Weight', 'maintain': 'Maintain Weight'}[x], label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back", key="s2b"): st.session_state.onboarding_step = 1; st.rerun()
        with c2:
            if st.button("Continue", key="s2"): st.session_state.user_profile['goal'] = goal; st.session_state.onboarding_step = 3; st.rerun()
    elif step == 3:
        st.markdown(f'<div style="text-align:center;margin-bottom:1.5rem;"><div class="icon-box orange" style="margin:0 auto 1rem;width:56px;height:56px;border-radius:16px;">{icon("activity", 28, "white")}</div><h3 style="margin:0;font-size:1.25rem;font-weight:700;">How active are you?</h3></div>', unsafe_allow_html=True)
        activity = st.radio("Activity:", ['sedentary', 'light', 'moderate', 'very_active'], format_func=lambda x: {'sedentary': 'Sedentary - Little exercise', 'light': 'Light - 1-3 days/week', 'moderate': 'Moderate - 3-5 days/week', 'very_active': 'Very Active - 6-7 days/week'}[x], label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back", key="s3b"): st.session_state.onboarding_step = 2; st.rerun()
        with c2:
            if st.button("Continue", key="s3"): st.session_state.user_profile['activity_level'] = activity; st.session_state.onboarding_step = 4; st.rerun()
    elif step == 4:
        st.markdown(f'<div style="text-align:center;margin-bottom:1.5rem;"><div class="icon-box pink" style="margin:0 auto 1rem;width:56px;height:56px;border-radius:16px;">{icon("calendar", 28, "white")}</div><h3 style="margin:0;font-size:1.25rem;font-weight:700;">Set your target</h3></div>', unsafe_allow_html=True)
        profile = st.session_state.user_profile
        cw, goal = profile.get('weight', 70), profile.get('goal', 'maintain')
        tw = st.number_input("Target Weight (kg)", 30.0 if goal == 'lose' else cw + 1, cw - 1 if goal == 'lose' else 200.0, cw - 5 if goal == 'lose' else cw + 5, 0.5) if goal != 'maintain' else cw
        timeline = st.slider("Timeline (weeks)", 4, 52, 12)
        daily_target = calculate_daily_calorie_target({**profile, 'target_weight': tw, 'timeline_weeks': timeline})
        st.markdown(f'<div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border-radius:16px;padding:1.5rem;margin:1.5rem 0;text-align:center;"><div style="font-size:0.75rem;opacity:0.9;text-transform:uppercase;">Daily Calorie Target</div><div style="font-size:2.5rem;font-weight:800;">{daily_target:,} cal</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back", key="s4b"): st.session_state.onboarding_step = 3; st.rerun()
        with c2:
            if st.button("Get Started", key="s4"): st.session_state.user_profile.update({'target_weight': tw, 'timeline_weeks': timeline}); st.session_state.onboarding_complete = True; st.rerun()

def render_dashboard():
    profile = st.session_state.user_profile
    daily_target = calculate_daily_calorie_target(profile)
    st.markdown(f'<div style="margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #e2e8f0;"><h1 style="margin:0;font-size:1.75rem;font-weight:700;color:#0f172a;">Welcome back, {profile.get("name", "User")}</h1><p style="margin:0;color:#64748b;font-size:0.875rem;">{datetime.now().strftime("%A, %B %d, %Y")}</p></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        remaining = max(0, daily_target - st.session_state.daily_calories_consumed)
        st.markdown(f'<div class="metric-card"><div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;"><div class="icon-box purple" style="width:44px;height:44px;">{icon("flame", 22, "white")}</div><div class="metric-label">Calories Remaining</div></div><div class="metric-value" style="color:#6366f1;">{remaining:,.0f}</div><div style="font-size:0.8125rem;color:#94a3b8;">of {daily_target:,} daily goal</div></div>', unsafe_allow_html=True)
        render_progress_bar(st.session_state.daily_calories_consumed, daily_target, 'progress-calories', 'Consumed')
    with c2:
        st.markdown(f'<div class="metric-card"><div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;"><div class="icon-box blue" style="width:44px;height:44px;">{icon("droplet", 22, "white")}</div><div class="metric-label">Water Intake</div></div><div class="metric-value" style="color:#0ea5e9;">{st.session_state.water_cups}/{RECOMMENDED_WATER_CUPS}</div><div style="font-size:0.8125rem;color:#94a3b8;">cups today</div></div>', unsafe_allow_html=True)
        render_progress_bar(st.session_state.water_cups, RECOMMENDED_WATER_CUPS, 'progress-water', 'Cups')
        wc1, wc2 = st.columns(2)
        with wc1:
            if st.button("Add Cup", key="aw", use_container_width=True): st.session_state.water_cups = min(st.session_state.water_cups + 1, 20); st.rerun()
        with wc2:
            if st.button("Remove", key="rw", use_container_width=True): st.session_state.water_cups = max(st.session_state.water_cups - 1, 0); st.rerun()
    with c3:
        bmi = calculate_bmi(profile['weight'], profile['height'])
        cat = get_bmi_category(bmi)
        color = "#f59e0b" if bmi < 18.5 else "#10b981" if bmi < 25 else "#f59e0b" if bmi < 30 else "#ef4444"
        st.markdown(f'<div class="metric-card"><div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;"><div class="icon-box green" style="width:44px;height:44px;">{icon("heart", 22, "white")}</div><div class="metric-label">Your BMI</div></div><div class="metric-value" style="color:{color};">{bmi:.1f}</div><div style="font-size:0.8125rem;color:#94a3b8;">{cat}</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{icon("activity", 20, "#6366f1")}<span>Today\'s Macros</span></div>', unsafe_allow_html=True)
    pt, ct, ft = daily_target * 0.25 / 4, daily_target * 0.50 / 4, daily_target * 0.25 / 9
    mc1, mc2, mc3 = st.columns(3)
    with mc1: render_progress_bar(st.session_state.daily_protein, pt, 'progress-protein', f'Protein ({st.session_state.daily_protein:.0f}g)')
    with mc2: render_progress_bar(st.session_state.daily_carbs, ct, 'progress-carbs', f'Carbs ({st.session_state.daily_carbs:.0f}g)')
    with mc3: render_progress_bar(st.session_state.daily_fat, ft, 'progress-fat', f'Fat ({st.session_state.daily_fat:.0f}g)')
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{icon("camera", 20, "#6366f1")}<span>Log Your Food</span></div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Camera / Upload", "Manual Entry"])
    
    with t1:
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            uploaded = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png'], key="fu", label_visibility="collapsed")
            camera = st.camera_input("Camera", key="fc", label_visibility="collapsed")
            image = Image.open(uploaded) if uploaded else Image.open(camera) if camera else None
            if image:
                st.image(image, use_container_width=True)
                meal_type = st.selectbox("Meal Type", ['breakfast', 'lunch', 'dinner', 'snack'], format_func=str.capitalize, key="mt")
                if st.button("Analyze Food", key="af", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        model, classes, err = load_model()
                        if model and not st.session_state.use_fallback: food, conf, _ = predict_food_local(model, classes, image)
                        else: food, conf, _ = predict_food_fallback(image)
                        if food:
                            st.session_state.prediction_result = {'food_name': food, 'confidence': conf}
                            st.session_state.nutrition_data = get_nutrition_data(food)
                            st.session_state.food_quantity_grams = 100  # Reset to default
                        st.session_state.use_fallback = False; st.rerun()
        with fc2:
            if st.session_state.prediction_result:
                r = st.session_state.prediction_result
                st.markdown(f'<div class="prediction-result"><div class="prediction-food">{r["food_name"].replace("_", " ").title()}</div><div class="prediction-confidence">Confidence: {r["confidence"]:.1%}</div></div>', unsafe_allow_html=True)
                if not st.session_state.use_fallback:
                    if st.button("Not correct? Try AI fallback", key="fb"): st.session_state.use_fallback = True; st.rerun()
                
                if st.session_state.nutrition_data:
                    n = st.session_state.nutrition_data
                    
                    # Quantity Slider
                    st.markdown(f'<div class="quantity-box"><div class="quantity-label">{icon("scale", 18, "#0369a1")} Portion Size</div></div>', unsafe_allow_html=True)
                    quantity = st.slider("Quantity (grams)", min_value=10, max_value=1500, value=st.session_state.food_quantity_grams, step=10, key="qty_slider", help="Adjust the portion size. Nutrition values are calculated per 100g by default.")
                    st.session_state.food_quantity_grams = quantity
                    
                    # Display quantity in kg if >= 1000g
                    if quantity >= 1000:
                        st.markdown(f'<p style="text-align:center;color:#0369a1;font-weight:600;margin-top:-0.5rem;">{quantity}g = {quantity/1000:.2f} kg</p>', unsafe_allow_html=True)
                    
                    # Get scaled nutrition values
                    scaled_n = get_scaled_nutrition(n, quantity)
                    
                    # Get AI advice with current quantity
                    if st.button("Get AI Health Advice", key="get_ai", use_container_width=True):
                        with st.spinner("Getting personalized advice..."):
                            st.session_state.ai_health_advice = get_ai_health_advice(r['food_name'], n, profile, quantity)
                        st.rerun()
                    
                    if st.session_state.ai_health_advice: 
                        render_ai_health_advice(st.session_state.ai_health_advice)
                    
                    # Nutrition Facts with scaled values
                    st.markdown(f'<div class="section-title" style="margin-top:1rem;">{icon("utensils", 18, "#6366f1")} Nutrition Facts <span style="font-weight:400;font-size:0.8rem;color:#64748b;">({quantity}g)</span></div>', unsafe_allow_html=True)
                    nc1, nc2, nc3 = st.columns(3)
                    with nc1: 
                        st.metric("Calories", f"{scaled_n.get('calories', 0):.0f}")
                        st.metric("Protein", f"{scaled_n.get('protein_g', 0):.1f}g")
                    with nc2: 
                        st.metric("Carbs", f"{scaled_n.get('carbohydrates_total_g', 0):.1f}g")
                        st.metric("Sugar", f"{scaled_n.get('sugar_g', 0):.1f}g")
                    with nc3: 
                        st.metric("Fat", f"{scaled_n.get('fat_total_g', 0):.1f}g")
                        st.metric("Fiber", f"{scaled_n.get('fiber_g', 0):.1f}g")
                    
                    if st.button("Add to Today's Log", key="al", use_container_width=True):
                        st.session_state.daily_calories_consumed += scaled_n.get('calories', 0)
                        st.session_state.daily_protein += scaled_n.get('protein_g', 0)
                        st.session_state.daily_carbs += scaled_n.get('carbohydrates_total_g', 0)
                        st.session_state.daily_fat += scaled_n.get('fat_total_g', 0)
                        st.session_state.meal_history.append({
                            'food_name': r['food_name'], 
                            'meal_type': st.session_state.get('mt', 'snack'), 
                            'calories': scaled_n.get('calories', 0), 
                            'protein': scaled_n.get('protein_g', 0),
                            'carbs': scaled_n.get('carbohydrates_total_g', 0), 
                            'fat': scaled_n.get('fat_total_g', 0), 
                            'quantity_grams': quantity,
                            'time': datetime.now().strftime('%I:%M %p')
                        })
                        st.session_state.prediction_result = None
                        st.session_state.nutrition_data = None
                        st.session_state.ai_health_advice = None
                        st.session_state.food_quantity_grams = 100
                        st.success(f"Added {quantity}g of {r['food_name'].replace('_', ' ').title()}!")
                        st.rerun()
            else:
                st.markdown(f'<div class="empty-state"><div style="color:#cbd5e1;margin-bottom:1rem;">{icon("upload", 48, "#cbd5e1")}</div><div style="font-weight:600;color:#64748b;">Upload a food image</div></div>', unsafe_allow_html=True)
    
    with t2:
        mf = st.text_input("Food name", placeholder="e.g., chicken salad")
        mm = st.selectbox("Meal", ['breakfast', 'lunch', 'dinner', 'snack'], format_func=str.capitalize, key="mm")
        if st.button("Get Nutrition Info", key="gn", use_container_width=True) and mf:
            with st.spinner("Fetching..."): n = get_nutrition_data(mf)
            if n: 
                st.session_state.nutrition_data = n
                st.session_state.prediction_result = {'food_name': mf, 'confidence': 1.0}
                st.session_state.food_quantity_grams = 100
                st.rerun()
            else: st.error("Could not find nutrition data.")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{icon("utensils", 20, "#6366f1")}<span>Today\'s Meals</span></div>', unsafe_allow_html=True)
    if not st.session_state.meal_history:
        st.markdown(f'<div class="empty-state"><div style="color:#cbd5e1;margin-bottom:1rem;">{icon("utensils", 48, "#cbd5e1")}</div><div style="font-weight:600;color:#64748b;">No meals logged yet</div></div>', unsafe_allow_html=True)
    else:
        for m in st.session_state.meal_history: render_meal_card(m)
    
    with st.sidebar:
        st.markdown('<h3 style="font-size:1rem;font-weight:700;margin-bottom:1.5rem;">Settings</h3>', unsafe_allow_html=True)
        if st.button("Reset Today's Data", use_container_width=True): reset_daily_data(); st.rerun()
        if st.button("Edit Profile", use_container_width=True): st.session_state.onboarding_complete = False; st.session_state.onboarding_step = 1; st.rerun()
        st.markdown('<p style="font-size:0.75rem;color:#94a3b8;text-align:center;margin-top:2rem;">FoodVisionAI v1.0<br>Data Analytics-3 Project</p>', unsafe_allow_html=True)

def main():
    load_css()
    init_session_state()
    if not st.session_state.onboarding_complete: render_onboarding()
    else: render_dashboard()

if __name__ == "__main__":
    main()
