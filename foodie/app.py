import os
import json
import requests
import uuid
import pickle
import secrets
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Flask
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Config
API_KEY = 'UR API KEY HERE'
BASE_URL = 'https://api.nal.usda.gov/fdc/v1/foods/search'
MODEL_DIR = 'models'
DIET_MODEL_PATH = os.path.join(MODEL_DIR, 'diet_model.pkl')
BINARY_MODEL_PATH = os.path.join(MODEL_DIR, 'binary_tasks_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Before request: assign user_id
@app.before_request
def assign_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

# Dummy data generator
def generate_dummy_data(samples=2000):
    np.random.seed(42)
    data = {
        'calories': np.random.randint(50, 800, samples),
        'fat': np.random.uniform(0, 40, samples).round(1),
        'carbs': np.random.uniform(0, 100, samples).round(1),
        'protein': np.random.uniform(0, 50, samples).round(1),
        'sugar': np.random.uniform(0, 50, samples).round(1),
        'fiber': np.random.uniform(0, 15, samples).round(1)
    }
    conditions = [
        (data['fat'] / data['calories'] > 0.3) & (data['protein'] < 20),
        (data['protein'] / data['calories'] > 0.3),
        (data['carbs'] / data['calories'] > 0.5),
    ]
    choices = ['high_fat', 'high_protein', 'high_carb']
    data['diet_type'] = np.select(conditions, choices, default='balanced')
    data['high_calorie'] = (data['calories'] > 400).astype(int)
    data['sugar_rich'] = (data['sugar'] > 20).astype(int)
    data['fiber_rich'] = (data['fiber'] > 5).astype(int)
    return pd.DataFrame(data)

def train_models():
    df = generate_dummy_data()
    X = df[['calories', 'fat', 'carbs', 'protein', 'sugar', 'fiber']]
    y_diet = df['diet_type']
    y_binary = df[['high_calorie', 'sugar_rich', 'fiber_rich']]

    X_train, X_test, y_train_d, y_test_d = train_test_split(X, y_diet, test_size=0.2, random_state=42)
    _, X_test_b, _, y_test_b = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Diet model
    diet_model = VotingClassifier([
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ], voting='soft')
    diet_model.fit(X_train_scaled, y_train_d)

    # Binary tasks
    multi_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    multi_clf.fit(X_train_scaled, y_binary.loc[X_train.index])

    # Save
    with open(DIET_MODEL_PATH, 'wb') as f: pickle.dump(diet_model, f)
    with open(BINARY_MODEL_PATH, 'wb') as f: pickle.dump(multi_clf, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

if not os.path.exists(DIET_MODEL_PATH) or not os.path.exists(BINARY_MODEL_PATH):
    train_models()

with open(DIET_MODEL_PATH, 'rb') as f: diet_model = pickle.load(f)
with open(BINARY_MODEL_PATH, 'rb') as f: binary_model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)

def get_nutrition_data(food_name):
    params = {'api_key': API_KEY, 'query': food_name, 'pageSize': 1}
    resp = requests.get(BASE_URL, params=params)
    if resp.status_code == 200:
        data = resp.json().get('foods', [])
        if data:
            item = data[0]
            nutrients = {n['nutrientName']: n['value'] for n in item.get('foodNutrients', [])}
            return {
                'name': item.get('description', food_name),
                'calories': nutrients.get('Energy', 0),
                'fat': nutrients.get('Total lipid (fat)', 0),
                'carbs': nutrients.get('Carbohydrate, by difference', 0),
                'protein': nutrients.get('Protein', 0),
                'sugar': nutrients.get('Sugars, total including NLEA', round(np.random.uniform(0, 50), 1)),
                'fiber': nutrients.get('Fiber, total dietary', round(np.random.uniform(0, 15), 1))
            }
    return {
        'name': food_name,
        'calories': np.random.randint(50, 800),
        'fat': round(np.random.uniform(0, 40), 1),
        'carbs': round(np.random.uniform(0, 100), 1),
        'protein': round(np.random.uniform(0, 50), 1),
        'sugar': round(np.random.uniform(0, 50), 1),
        'fiber': round(np.random.uniform(0, 15), 1)
    }
    

def analyze_nutrition(foods):
    totals = {'calories': 0, 'fat': 0, 'carbs': 0, 'protein': 0, 'sugar': 0, 'fiber': 0, 'foods': [], 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    for food in foods:
        d = get_nutrition_data(food)
        totals['foods'].append(d)
        for key in ['calories', 'fat', 'carbs', 'protein', 'sugar', 'fiber']:
            totals[key] += d[key]
    X = scaler.transform([[totals[k] for k in ['calories','fat','carbs','protein','sugar','fiber']]])
    totals['diet_type'] = diet_model.predict(X)[0]
    b_preds = binary_model.predict(X)[0]
    totals['high_calorie'] = bool(b_preds[0])
    totals['sugar_rich'] = bool(b_preds[1])
    totals['fiber_rich'] = bool(b_preds[2])
    return totals

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['selected_foods'] = request.form.getlist('food_name[]')
        return redirect(url_for('results'))
    session.setdefault('selected_foods', [''])
    return render_template('index.html', foods=session['selected_foods'])

@app.route('/add-food', methods=['POST'])
def add_food():
    session['selected_foods'].append('')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    selected = [f for f in session.get('selected_foods', []) if f.strip()]
    if not selected:
        flash('Please enter at least one food item.', 'warning')
        return redirect(url_for('index'))
    nutrition = analyze_nutrition(selected)
    db.collection("users").document(session['user_id']).collection("history").add(nutrition)
    return render_template('results.html',
        nutrition=nutrition,
        diet_type=nutrition['diet_type'].replace('_', ' ').title(),
        high_cal='Yes' if nutrition['high_calorie'] else 'No',
        sugar_rich='Yes' if nutrition['sugar_rich'] else 'No',
        fiber_rich='Yes' if nutrition['fiber_rich'] else 'No'
    )

@app.route('/history')
def history():
    user_id = session['user_id']
    docs = db.collection("users").document(user_id).collection("history").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    history = [doc.to_dict() for doc in docs]
    return render_template('history1.html', history=history)

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
