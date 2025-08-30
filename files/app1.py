from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import sqlite3, hashlib, os, pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "your-secret-key"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- DATABASE CONNECTION ----------

def get_db_connection():
    return sqlite3.connect('users.db')

# ---------- LOAD MODEL & DATA ----------

with open('model.pkl', 'rb') as f:
    scaler = pickle.load(f)
    rfc = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Crop details 
crop_details = {
    "rice": {
        "info": "Rice is a staple food crop that requires warm weather and plenty of water.",
        "image": "rice.jpg",
        "soil": "Clayey loam to loam",
        "season": "Kharif (June–November)",
        "temperature": "20°C to 37°C",
        "water": "High (requires standing water)",
        "climate": "Tropical and subtropical"
    },
    "maize": {
        "info": "Maize grows well in fertile, well-drained soil with moderate rainfall.",
        "image": "maize.jpg",
        "soil": "Well-drained loamy soil",
        "season": "Kharif and Rabi",
        "temperature": "21°C to 27°C",
        "water": "Moderate",
        "climate": "Subtropical"
    },
    "chickpea": {
        "info": "Chickpeas prefer dry climates and are rich in protein.",
        "image": "chickpea.jpg",
        "soil": "Well-drained sandy loam",
        "season": "Rabi (October–March)",
        "temperature": "10°C to 25°C",
        "water": "Low",
        "climate": "Cool and dry"
    },
    "kidneybeans": {
        "info": "Kidney beans need warm weather and moderately fertile soil.",
        "image": "kidneybeans.jpg",
        "soil": "Loamy soil with good drainage",
        "season": "Kharif",
        "temperature": "18°C to 27°C",
        "water": "Moderate",
        "climate": "Tropical and temperate"
    },
    "pigeonpeas": {
        "info": "Pigeon peas grow best in tropical and subtropical climates.",
        "image": "pigeonpeas.jpg",
        "soil": "Loamy or black cotton soil",
        "season": "Kharif",
        "temperature": "26°C to 30°C",
        "water": "Moderate",
        "climate": "Semi-arid to sub-humid"
    },
    "mothbeans": {
        "info": "Moth beans are drought-resistant legumes grown in arid regions.",
        "image": "mothbeans.jpg",
        "soil": "Sandy loam",
        "season": "Kharif",
        "temperature": "25°C to 35°C",
        "water": "Very low",
        "climate": "Hot and dry"
    },
    "mungbean": {
        "info": "Mungbeans are rich in nutrients and grow well in warm climates.",
        "image": "mungbean.jpg",
        "soil": "Well-drained loamy soil",
        "season": "Kharif and Summer",
        "temperature": "25°C to 35°C",
        "water": "Low to moderate",
        "climate": "Warm and humid"
    },
    "blackgram": {
        "info": "Black gram is a leguminous crop ideal for loamy soil.",
        "image": "blackgram.jpg",
        "soil": "Fertile loamy soil",
        "season": "Kharif and Rabi",
        "temperature": "25°C to 35°C",
        "water": "Moderate",
        "climate": "Subtropical"
    },
    "lentil": {
        "info": "Lentils require a cool growing season and well-drained soil.",
        "image": "lentil.jpg",
        "soil": "Loamy and clay loam",
        "season": "Rabi (Winter)",
        "temperature": "15°C to 25°C",
        "water": "Low",
        "climate": "Cool and dry"
    },
    "banana": {
        "info": "Bananas need rich soil and warm, humid climates.",
        "image": "banana.jpg",
        "soil": "Fertile loamy soil",
        "season": "Year-round",
        "temperature": "26°C to 30°C",
        "water": "High",
        "climate": "Tropical"
    },
    "apple": {
        "info": "Apples grow best in cool climates with loamy soil.",
        "image": "apple.jpg",
        "soil": "Loamy, well-drained",
        "season": "Autumn",
        "temperature": "15°C to 21°C",
        "water": "Moderate",
        "climate": "Temperate"
    },
    "cotton": {
        "info": "Cotton grows well in warm climates with adequate sunshine.",
        "image": "cotton.jpg",
        "soil": "Black cotton soil",
        "season": "Kharif",
        "temperature": "21°C to 30°C",
        "water": "Moderate",
        "climate": "Warm and dry"
    },
    "coffee": {
        "info": "Coffee grows best in tropical climates at higher altitudes.",
        "image": "coffee.jpg",
        "soil": "Well-drained rich soil",
        "season": "Winter to early summer",
        "temperature": "15°C to 28°C",
        "water": "Moderate",
        "climate": "Tropical highland"
    },
    "grapes": {
        "info": "Grapes prefer warm days and cool nights in well-drained soil.",
        "image": "grapes.jpg",
        "soil": "Sandy to clay loam",
        "season": "Spring",
        "temperature": "15°C to 30°C",
        "water": "Moderate",
        "climate": "Temperate and subtropical"
    },
    "watermelon": {
        "info": "Watermelons thrive in sandy loam soil with good sunlight.",
        "image": "watermelon.jpg",
        "soil": "Sandy loam",
        "season": "Summer",
        "temperature": "24°C to 30°C",
        "water": "High",
        "climate": "Warm"
    },
    "muskmelon": {
        "info": "Muskmelons grow in sandy, well-aerated soil with good drainage.",
        "image": "muskmelon.jpg",
        "soil": "Sandy soil",
        "season": "Summer",
        "temperature": "24°C to 30°C",
        "water": "Moderate",
        "climate": "Warm"
    },
    "orange": {
        "info": "Oranges need a warm, subtropical climate and plenty of sunlight.",
        "image": "orange.jpg",
        "soil": "Sandy loam",
        "season": "Winter",
        "temperature": "13°C to 37°C",
        "water": "Moderate",
        "climate": "Subtropical"
    },
    "papaya": {
        "info": "Papayas require tropical weather and well-drained soil.",
        "image": "papaya.jpg",
        "soil": "Loamy soil",
        "season": "Throughout the year",
        "temperature": "21°C to 33°C",
        "water": "Moderate",
        "climate": "Tropical"
    },
    "coconut": {
        "info": "Coconut palms grow in coastal areas with sandy soil.",
        "image": "coconut.jpg",
        "soil": "Sandy soil",
        "season": "Year-round",
        "temperature": "27°C to 32°C",
        "water": "High",
        "climate": "Tropical coastal"
    },
    "jute": {
        "info": "Jute thrives in alluvial soil and humid climates.",
        "image": "jute.jpg",
        "soil": "Alluvial soil",
        "season": "Kharif",
        "temperature": "24°C to 37°C",
        "water": "High",
        "climate": "Humid"
    }
}

@app.route('/')
def landing():
    return redirect('/register')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        father_name = request.form['father_name']
        dob = request.form['dob']
        aadhaar = request.form['aadhaar']
        phone = request.form['phone']
        photo_file = request.files['photo']

        if password != confirm_password:
            return render_template('register.html', msg="Passwords do not match!")

        if len(password) < 8:
            return render_template('register.html', msg="Password too weak!")

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        photo_filename = ""
        if photo_file and photo_file.filename != "":
            photo_filename = os.path.join(UPLOAD_FOLDER, photo_file.filename)
            photo_file.save(photo_filename)

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        if c.fetchone():
            conn.close()
            return render_template('register.html', msg="Email already registered.")

        c.execute("INSERT INTO users (username, email, password, father_name, dob, aadhaar, phone, photo) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (username, email, hashed_password, father_name, dob, aadhaar, phone, photo_filename))
        conn.commit()
        conn.close()
        return redirect('/login')

    return render_template('register.html', msg='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect('/recommend')
        else:
            return render_template('login.html', msg="Invalid email or password.")

    return render_template('login.html', msg='')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        try:
            N = int(request.form['Nitrogen'])
            P = int(request.form['Phosphorus'])
            K = int(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['pH'])
            rainfall = float(request.form['Rainfall'])

            if not (0 <= N <= 140 and 5 <= P <= 145 and 5 <= K <= 205 and
                    8 <= temp <= 45 and 10 <= humidity <= 100 and
                    3.5 <= ph <= 9 and 20 <= rainfall <= 300):
                return render_template('index.html',
                    result="Input values are out of range.",
                    crop_info="N/A",
                    crop_image="default.jpg",
                    crop_soil="N/A",
                    crop_season="N/A",
                    crop_temp="N/A",
                    crop_water="N/A",
                    crop_climate="N/A")

            data = [N, P, K, temp, humidity, ph, rainfall]
            features = scaler.transform([data])
            prediction = rfc.predict(features)
            predicted_crop = le.inverse_transform(prediction)[0]

            details = crop_details.get(predicted_crop, {
                "info": "No information available.",
                "image": "default.jpg",
                "soil": "Not specified",
                "season": "Not specified",
                "temperature": "Not specified",
                "water": "Not specified",
                "climate": "Not specified"
            })

            return render_template('index.html',
                result=predicted_crop,
                crop_info=details['info'],
                crop_image=details['image'],
                crop_soil=details['soil'],
                crop_season=details['season'],
                crop_temp=details['temperature'],
                crop_water=details['water'],
                crop_climate=details['climate'])

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html',
                result="Invalid input!",
                crop_info="N/A",
                crop_image="default.jpg",
                crop_soil="N/A",
                crop_season="N/A",
                crop_temp="N/A",
                crop_water="N/A",
                crop_climate="N/A")

    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
