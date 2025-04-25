from flask import Flask, request, render_template, jsonify  # Import Flask web framework and related modules
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import pickle  # Import pickle for loading the ML model
from fuzzywuzzy import process  # Import fuzzy_wuzzy for string matching
import ast  # Import ast for parsing string representations of Python literals
from datetime import datetime  # Import datetime for timestamp in PDF report

app = Flask(__name__)  # Initialize Flask application

# Loading the datasets
sym_des = pd.read_csv("Dataset/symptoms_df.csv") 
precautions = pd.read_csv("Dataset/precautions_df.csv")  
workout = pd.read_csv("Dataset/workout_df.csv") 
description = pd.read_csv("Dataset/description.csv")  
medications = pd.read_csv('Dataset/medications.csv')  
diets = pd.read_csv("Dataset/diets.csv")  

Rf = pickle.load(open('model/RandomForest.pkl','rb')) 

# Here we make a dictionary of symptoms and diseases and preprocess it
symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}  # Dictionary mapping symptoms to numerical indices for model input
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}  # Dictionary mapping numerical disease indices to disease names

symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()} 


def information(predicted_dis):  # Function to gather all information about a predicted disease
    disease_desciption = description[description['Disease'] == predicted_dis]['Description']  
    disease_desciption = " ".join([w for w in disease_desciption])  # Join description into a single string

    disease_precautions = precautions[precautions['Disease'] == predicted_dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]  # Get precautions
    disease_precautions = [col for col in disease_precautions.values] 

    disease_medications = medications[medications['Disease'] == predicted_dis]['Medication']  # Get medications
    disease_medications = [med for med in disease_medications.values]  

    disease_diet = diets[diets['Disease'] == predicted_dis]['Diet']  # Get dietary recommendations
    disease_diet = [die for die in disease_diet.values]  # Convert to list

    disease_workout = workout[workout['disease'] == predicted_dis] ['workout']  # Get workout recommendations
 
    return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout  # Return all information

# user input symptoms to our Model
def predicted_value(patient_symptoms):  
    i_vector = np.zeros(len(symptoms_list_processed))  
    for i in patient_symptoms:  
        i_vector[symptoms_list_processed[i]] = 1  
    return diseases_list[Rf.predict([i_vector])[0]] 

# Function to correct the spellings of the symptom (if any)
def correct_spelling(symptom):  
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys()) 
    if score >= 80: 
        return closest_match  
    else:
        return None  

@app.route('/predict', methods=['GET', 'POST'])  # Define route for prediction endpoint
def home():  # Function to handle prediction requests
    if request.method == 'POST':  # If the request is a POST request
        symptoms = request.form.get('symptoms')  # Get symptoms from form data
        if symptoms == "Symptoms":  # If default placeholder text is submitted
            message = "Please either write symptoms or you have written misspelled symptoms"  # Error message
            return render_template('index.html', message=message)  # Return template with error message
        else:
           
            patient_symptoms = [s.strip() for s in symptoms.split(',')]  # Split by comma and remove whitespace
            # Remove any extra characters, if any
            patient_symptoms = [symptom.strip("[]' ") for symptom in patient_symptoms]  # Clean up symptom strings

            # Correct the spelling of symptoms
            corrected_symptoms = []  # Initialize empty list for corrected symptoms
            for symptom in patient_symptoms:  # For each symptom entered by the user
                corrected_symptom = correct_spelling(symptom)  # Try to correct the spelling
                if corrected_symptom:  # If a match was found
                    corrected_symptoms.append(corrected_symptom)  # Add to list of corrected symptoms
                else:
                    message = f"Symptom '{symptom}' not found in the database."  # Error message for unknown symptom
                    return render_template('index.html', message=message)  # Return template with error message

            # Predict the disease using corrected symptoms
            predicted_disease = predicted_value(corrected_symptoms)  # Predict disease based on symptoms
            dis_des, precautions, medications, rec_diet, workout = information(predicted_disease)  # Get information about the disease

            my_precautions = []  # Initialize empty list for precautions
            for i in precautions[0]:  # For each precaution
                my_precautions.append(i)  # Add to list
            
            #converting the string into a list format before returning to the frontend
            medication_list = ast.literal_eval(medications[0])  # Convert string representation to Python list
            medications = []  # Initialize empty list
            for item in medication_list:  # For each medication
                medications.append(item)  # Add to list
            
            diet_list = ast.literal_eval(rec_diet[0])  # Convert string representation to Python list
            rec_diet = []  # Initialize empty list
            for item in diet_list:  # For each diet recommendation
                rec_diet.append(item)  # Add to list
            return render_template('index.html', symptoms=corrected_symptoms, predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)  # Return template with all disease information

    return render_template('index.html')  # For GET requests, just render the template

# about view function and path
@app.route('/')  # Define route for home page
def index():  # Function to handle home page requests
    return render_template('index.html')  # Render the index template

@app.route('/emergency_card', methods=['GET'])
def emergency_card():
    """Route to display the emergency card form"""
    # Get any disease prediction data from the session if available
    predicted_disease = request.args.get('disease', None)
    medications = request.args.getlist('medications')
    
    return render_template('emergency_card.html', 
                          predicted_disease=predicted_disease,
                          medications=medications)

@app.route('/generate_emergency_card', methods=['POST'])
def generate_emergency_card():
    """Route to generate the emergency information card"""
    # Get all form data
    form_data = request.form.to_dict()
    
    # Format the date of birth to be more readable
    if form_data.get('dateOfBirth'):
        dob = datetime.strptime(form_data['dateOfBirth'], '%Y-%m-%d')
        form_data['dateOfBirth'] = dob.strftime('%b %d, %Y')
    
    # Add creation date
    form_data['creationDate'] = datetime.now().strftime('%b %d, %Y')
    
    # Return the template with the emergency card data
    return render_template('emergency_card.html', emergency_card=form_data)

# Add a route to link from the prediction page to the emergency card page
@app.route('/create_emergency_card', methods=['POST'])
def create_emergency_card():
    """Route to redirect from prediction to emergency card form with data"""
    predicted_disease = request.form.get('predicted_disease', '')
    medications = request.form.getlist('medications')
    
    # Redirect to emergency card page with the disease and medications as query parameters
    return render_template('emergency_card.html', 
                          predicted_disease=predicted_disease,
                          medications=medications)

@app.route('/view_report', methods=['POST'])
def view_report():
    # Get the data from the form
    symptoms = request.form.get('symptoms', '')
    predicted_disease = request.form.get('predicted_disease', '')
    dis_des = request.form.get('dis_des', '')
    
    # Get the lists from form
    my_precautions = request.form.getlist('my_precautions')
    medications = request.form.getlist('medications')
    my_diet = request.form.getlist('my_diet')
    workout = request.form.getlist('workout')
    
    # Get current datetime for the report
    now = datetime.now()
    
    # Render the report template with the data
    return render_template('report.html', 
                          symptoms=symptoms,
                          predicted_disease=predicted_disease, 
                          dis_des=dis_des,
                          my_precautions=my_precautions, 
                          medications=medications, 
                          my_diet=my_diet,
                          workout=workout,
                          now=now)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    # Get the data from the form
    symptoms = request.form.get('symptoms', '')
    predicted_disease = request.form.get('predicted_disease', '')
    dis_des = request.form.get('dis_des', '')
    
    # Get the lists from form
    my_precautions = request.form.getlist('my_precautions')
    medications = request.form.getlist('medications')
    my_diet = request.form.getlist('my_diet')
    workout = request.form.getlist('workout')
    
    # Get current datetime for the report
    now = datetime.now()
    
    # Create a report data object
    report_data = {
        'symptoms': symptoms,
        'predicted_disease': predicted_disease,
        'dis_des': dis_des,
        'my_precautions': my_precautions,
        'medications': medications,
        'my_diet': my_diet,
        'workout': workout,
        'generated_date': now.strftime('%B %d, %Y at %I:%M %p')
    }
    
    # Render the report template with the data
    return render_template('report_download.html', report=report_data)

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask application in debug mode