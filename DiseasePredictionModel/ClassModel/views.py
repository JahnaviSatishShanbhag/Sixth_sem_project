from django.shortcuts import render
import joblib

def home(request):
    return render(request,"home.html")

def result(request):
    cls_dt=joblib.load('finalized_model.sav')
    cls_knn=joblib.load('finalized_model_knn.sav')
    cls_nb=joblib.load('finalized_model_nb.sav')
    cls_rf=joblib.load('finalized_model_rf.sav')
    lis=[]
    lis.append(request.GET['Symptom1'])
    lis.append(request.GET['Symptom2'])
    lis.append(request.GET['Symptom3'])
    lis.append(request.GET['Symptom4'])
    lis.append(request.GET['Symptom5'])

    l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
          'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
          'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
          'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
          'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
          'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
          'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
          'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
          'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
          'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
          'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
          'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
          'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
          'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
          'family_history', 'mucoid_sputum',
          'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
          'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
          'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
          'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
          'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
          'yellow_crust_ooze']

    disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
               'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
               'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
               'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
               'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
               'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
               'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
               'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
               'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
               'Osteoarthristis', 'Arthritis',
               '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
               'Urinary tract infection', 'Psoriasis', 'Impetigo']

    l2 = []
    for i in range(0, len(l1)):
        l2.append(0)

    for k in range(0, len(l1)):
        for z in lis:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]

    # Decision tree
    predict_dt = cls_dt.predict(inputtest)
    predicted_dt = predict_dt[0]
    # print('Predicted', predicted)

    # KNN
    predict_knn = cls_knn.predict(inputtest)
    predicted_knn = predict_knn[0]

    # Naive Bayes
    predict_nb = cls_nb.predict(inputtest)
    predicted_nb = predict_nb[0]

    # Random forest
    predict_rf = cls_rf.predict(inputtest)
    predicted_rf = predict_rf[0]

    # is_disease_found = 'no'
    # for a in range(0, len(disease)):
    #     if (predicted_dt == a):
    #         is_disease_found = 'yes'
    #         break

    # if (is_disease_found == 'yes'):
    #     ans = disease[predicted_dt]
    #     context = {'disease_dt': ans}
    #     return render(request, 'result.html', context)

    # else:
    #     print(lis)
    #     print("Disease Not Found")
    #     ans = "No Disease Found"
    #     context = {'disease_dt': ans}
    #     return render(request, 'result.html', context)

    ans_dt = disease[predicted_dt]
    ans_knn = disease[predicted_knn]
    ans_nb = disease[predicted_nb]
    ans_rf = disease[predicted_rf]
    
    context = {'disease_dt': ans_dt,'disease_knn':ans_knn,'disease_nb':ans_nb,'disease_rf':ans_rf}

    return render(request, 'result.html', context)

def diseasePrediction(request):
    return render(request,"disease_prediction.html")