from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv(r'C:\Users\preethi\Downloads\insurance_data.csv')

# Preprocess data
X = df.drop('claim', axis=1)
y = df['claim']

categorical_features = ['gender', 'medical_history', 'occupation', 'lifestyle_habits', 'city']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), ['age'])
    ]
)

X_processed = preprocessor.fit_transform(X)

# Handle class imbalance using under-sampling
under = RandomUnderSampler(sampling_strategy='auto')
X_resampled, y_resampled = under.fit_resample(X_processed, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train the model
model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Define mappings for occupation and city
occupation_dict = {
    1: "Low Risk: Student,Teacher, Accountant, Software Developer, Librarian, Graphic Designer, Writer, Data Analyst, Receptionist, HR Specialist, Web Developer, Pharmacist, Research Scientist, Financial Analyst",
    2: "Moderate Risk: Construction Worker, Electrician, Mechanic, Carpenter, Plumber, Chef, Nurse, Truck Driver, Warehouse Worker, Delivery Driver, Fitness Trainer, Bartender, Event Planner",
    3: "High Risk: Firefighter, Police Officer, Pilot, Miner, Roofer, Oil Rig Worker, Crane Operator, Fisherman, Soldier, Window Cleaner (High Rise), Logger, Welder, Scaffolder"
}

city_dict = {
    1: "Tier 1: Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad",
    2: "Tier 2: Surat, Jaipur, Lucknow, Kanpur, Nagpur, Visakhapatnam, Bhopal, Patna, Vadodara, Indore, Thane, Agra, Nashik, Faridabad, Meerut",
    3: "Tier 3: Jabalpur, Coimbatore, Guwahati, Gwalior, Vijayawada, Mysore, Ranchi, Raipur, Kochi, Kozhikode, Hubli-Dharwad, Belgaum, Jammu, Jodhpur, Tiruchirappalli, Bareilly"
}

# Reverse dictionaries for lookup
inverse_occupation_dict = {v: k for k, v in occupation_dict.items()}
inverse_city_dict = {v: k for k, v in city_dict.items()}

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    probability_claim = None
    probability_no_claim = None
    age = None
    gender = None
    medical_history_str = ''
    occupation_name = None
    lifestyle_habits_str = ''
    city_name = None

    if request.method == 'POST':
        try:
            age = request.form.get('age')
            gender = request.form.get('gender')
            medical_history = request.form.getlist('medical_history')
            occupation_name = request.form.get('occupation')
            lifestyle_habits = request.form.getlist('lifestyle_habits')
            city_name = request.form.get('city')

            age = int(age) if age else None
            occupation = inverse_occupation_dict.get(occupation_name, None)
            city_tier = inverse_city_dict.get(city_name, None)
            medical_history_str = ', '.join(medical_history) if medical_history else ''
            lifestyle_habits_str = ', '.join(lifestyle_habits) if lifestyle_habits else ''

            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'medical_history': [medical_history_str],
                'occupation': [occupation],
                'lifestyle_habits': [lifestyle_habits_str],
                'city': [city_name]
            })

            input_processed = preprocessor.transform(input_data)
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            probability_claim = probabilities[1]
            probability_no_claim = probabilities[0]

        except Exception as e:
            print(f"Error processing request: {e}")

    return render_template('index.html', 
                           prediction=prediction, 
                           probability_claim=probability_claim,
                           probability_no_claim=probability_no_claim,
                           age=age,
                           gender=gender,
                           medical_history=medical_history_str,
                           occupation_name=occupation_name,
                           lifestyle_habits=lifestyle_habits_str,
                           city_name=city_name)
if __name__ == '__main__':
    app.run(debug=True)
