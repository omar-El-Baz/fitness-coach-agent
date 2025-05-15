import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import OneHotEncoder # For recreating input structure if needed

# --- Configuration and Constants ---
USER_PROFILES_DB = 'user_profiles.csv'
MODEL_PATH = 'diet_model.pkl'
GYM_DATA_PATH = 'gym_recommendation_data.csv' # Mendeley dataset
DIET_RAW_DATA_PATH = 'diet_recommendations.csv' # Kaggle dataset for dropdowns

# --- Load Models and Data ---
@st.cache_resource # Use cache_resource for models/data that don't change
def load_diet_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Diet model not found at {MODEL_PATH}. Please run diet_model_trainer.py first.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading diet model: {e}")
        return None

@st.cache_data # Use cache_data for dataframes from CSVs
def load_gym_data():
    if not os.path.exists(GYM_DATA_PATH):
        st.error(f"Gym recommendation data not found at {GYM_DATA_PATH}.")
        return pd.DataFrame() # Return empty DataFrame
    try:
        # Try reading as CSV, then Excel if CSV fails (common for Mendeley)
        try:
            df = pd.read_csv(GYM_DATA_PATH)
        except pd.errors.ParserError: # Or other specific CSV errors
            df = pd.read_excel(GYM_DATA_PATH)
        
        # Basic cleaning specific to gym_recommendation_data
        df.columns = df.columns.str.strip().str.replace('\s+', '_', regex=True) # Clean column names
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].str.capitalize()
        if 'Fitness_Goal' in df.columns:
            df['Fitness_Goal'] = df['Fitness_Goal'].str.strip()

        return df
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        st.error(f"File not found: {GYM_DATA_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading gym data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_diet_raw_data_for_options():
    if not os.path.exists(DIET_RAW_DATA_PATH):
        st.error(f"Raw diet data for options not found at {DIET_RAW_DATA_PATH}.")
        return pd.DataFrame()
    try:
        return pd.read_csv(DIET_RAW_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading raw diet data: {e}")
        return pd.DataFrame()

diet_model_pipeline = load_diet_model()
gym_df = load_gym_data()
diet_raw_df = load_diet_raw_data_for_options()

# --- Helper Functions ---
def calculate_bmi(weight, height_cm):
    if height_cm == 0: return 0
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif 18.5 <= bmi < 24.9: return "Normal weight"
    elif 25 <= bmi < 29.9: return "Overweight"
    else: return "Obese"

def save_user_profile(user_data):
    try:
        if os.path.exists(USER_PROFILES_DB):
            profiles_df = pd.read_csv(USER_PROFILES_DB)
            # Create a DataFrame from user_data dictionary
            new_profile_df = pd.DataFrame([user_data])
            profiles_df = pd.concat([profiles_df, new_profile_df], ignore_index=True)
        else:
            profiles_df = pd.DataFrame([user_data])
        profiles_df.to_csv(USER_PROFILES_DB, index=False)
        st.sidebar.success("Profile saved!")
    except Exception as e:
        st.sidebar.error(f"Error saving profile: {e}")

def get_workout_recommendation(user_profile, gym_data):
    if gym_data.empty:
        return {"error": "Gym data not available for workout recommendation."}

    goal = user_profile['fitness_goal']
    user_bmi_category = user_profile['bmi_category'] # e.g., Normal weight, Overweight
    user_sex = user_profile['gender'] # Male/Female
    has_hypertension = user_profile['hypertension']
    has_diabetes = user_profile['diabetes']

    # Filter based on Fitness Goal (primary)
    filtered_df = gym_data[gym_data['Fitness_Goal'].str.contains(goal, case=False, na=False)].copy()

    if filtered_df.empty:
        return {"error": f"No specific plan found for goal: {goal}. Showing general advice if available."}

    # Optional: Further filter by BMI Level, Sex, Health Conditions if data allows and is rich enough
    # For simplicity, we'll make these softer preferences or pick the most common if goal-specific plans exist.
    
    # Example: Prefer plans matching sex if available
    sex_match_df = filtered_df[filtered_df['Sex'].str.contains(user_sex, case=False, na=False)]
    if not sex_match_df.empty:
        filtered_df = sex_match_df
    
    # Example: Prefer plans matching BMI Level if available (map gym_df 'Level' to our categories)
    # This requires mapping 'Level' in gym_df to 'Underweight', 'Normal weight', etc.
    # For now, we'll skip this detailed BMI level matching for simplicity and use the broader fitness goal.

    # Prioritize plans that align with health conditions
    # If user has hypertension, prefer plans where Hypertension is 'No' or not specified as risky
    # (Assuming 'Hypertension' column in gym_df indicates if plan is SUITABLE for people WITH hypertension, or if it's a user attribute)
    # The Mendeley dataset's 'Hypertension'/'Diabetes' columns describe the *user*, not the plan's suitability.
    # So, we might look for plans recommended for users with similar conditions or general safe plans.
    
    # For simplicity, let's pick the first matching plan for the given goal.
    # A more robust system would rank or provide options.
    if not filtered_df.empty:
        rec = filtered_df.iloc[0].to_dict()
        workout_plan = {
            "Fitness Type": rec.get("Fitness_Type", "Not specified"),
            "Exercises": rec.get("Exercises", "General exercises suitable for the goal."),
            "Equipment": rec.get("Equipment", "Basic equipment / Bodyweight."),
            "General Diet Advice (from workout plan)": rec.get("Diet", "Follow a balanced diet."),
            "General Recommendation": rec.get("Recommendation", "Stay consistent and listen to your body.")
        }
        return workout_plan
    else:
        return {"error": f"Could not find a sufficiently matching workout for {goal} and other criteria."}


def get_diet_prediction(user_profile, model_pipeline, raw_diet_data):
    if model_pipeline is None:
        return "Diet model not loaded."
    if raw_diet_data.empty:
        return "Raw diet data for options not available."

    # Prepare input for the diet model based on expected features during training
    # This must exactly match the structure and encoding of X_train in diet_model_trainer.py
    
    disease_type = "None"
    if user_profile['hypertension'] and user_profile['diabetes']:
        disease_type = "Diabetes_Hypertension" # Or handle as two separate conditions if model trained that way
                                              # For simplicity, let's map to the closest from diet_raw_df
                                              # Assuming model trained on 'Diabetes', 'Hypertension', 'Obesity', 'None'
        if "Diabetes" in raw_diet_data['Disease_Type'].unique(): disease_type = "Diabetes" # Prioritize
        elif "Hypertension" in raw_diet_data['Disease_Type'].unique(): disease_type = "Hypertension"

    elif user_profile['hypertension']:
        disease_type = "Hypertension"
    elif user_profile['diabetes']:
        disease_type = "Diabetes"
    
    # If BMI indicates obesity, set disease_type to Obesity if not already set by other conditions
    # This aligns with how 'Disease_Type' in Dataset 2 might be used.
    if user_profile['bmi_category'] == "Obese" and disease_type == "None":
        if "Obesity" in raw_diet_data['Disease_Type'].unique():
             disease_type = "Obesity"


    input_data = {
        'Age': user_profile['age'],
        'Gender': user_profile['gender'],
        'Weight_kg': user_profile['weight'],
        'Height_cm': user_profile['height'],
        'BMI_calculated': user_profile['bmi'],
        'Disease_Type': disease_type,
        'Physical_Activity_Level': user_profile['activity_level'],
        'Dietary_Restrictions': user_profile['restrictions'],
        'Allergies': ", ".join(user_profile['allergies']) if user_profile['allergies'] else "None" # Model expects string
    }
    
    # Ensure 'Allergies' is one of the trained categories or 'None'
    # The OneHotEncoder (handle_unknown='ignore') will manage new allergy strings not seen during training.
    
    # Ensure 'Disease_Type' is one of the trained categories, or map to 'None'
    trained_disease_types = raw_diet_data['Disease_Type'].unique()
    if input_data['Disease_Type'] not in trained_disease_types:
        input_data['Disease_Type'] = "None" # Default if not a recognized disease type

    input_df = pd.DataFrame([input_data])
    
    # Ensure column order and types match what the preprocessor in the pipeline expects
    # The pipeline's ColumnTransformer handles this if column names are consistent.
    # Let's make sure we have all columns the preprocessor expects from `features_for_diet_model`
    # in `diet_model_trainer.py`.
    
    # The pipeline's preprocessor expects columns in the order they were defined
    # during training. If diet_model_trainer.py used X = diet_df[features_for_diet_model],
    # then input_df should also have columns in that order.
    
    expected_cols_in_order = [ # From diet_model_trainer.py
        'Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI_calculated',
        'Disease_Type', 'Physical_Activity_Level',
        'Dietary_Restrictions', 'Allergies'
    ]
    input_df = input_df[expected_cols_in_order]


    try:
        prediction = model_pipeline.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during diet prediction: {e}")
        # You might want to inspect `input_df` here or how the preprocessor handles its columns
        st.error(f"Input data structure for model: {input_df.to_dict()}")
        # Access the preprocessor to see its transformers and expected features
        # preprocessor_step = model_pipeline.named_steps['preprocessor']
        # st.write("Preprocessor transformers:", preprocessor_step.transformers_)
        return "Error in prediction"

# --- Streamlit UI ---
st.set_page_config(page_title="Fitness Coach Agent", layout="wide")
st.title("💪 Personalized Fitness Coach Agent 🏋️‍♀️🥗")

# --- User Input Sidebar ---
st.sidebar.header("👤 Your Profile")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
gender_options = ["Male", "Female"]
if not gym_df.empty and 'Sex' in gym_df.columns: # Use actual values if available
    gender_options = list(gym_df['Sex'].str.capitalize().unique())
    # Ensure Male/Female are present for diet model compatibility
    if "Male" not in gender_options: gender_options.append("Male")
    if "Female" not in gender_options: gender_options.append("Female")
    gender_options = sorted(list(set(gender_options))) # Unique and sorted
gender = st.sidebar.selectbox("Gender", options=gender_options, index=0)

weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)

st.sidebar.subheader("🩺 Health Conditions")
hypertension = st.sidebar.checkbox("Hypertension (High Blood Pressure)")
diabetes = st.sidebar.checkbox("Diabetes")

st.sidebar.subheader("🎯 Fitness Goals & Preferences")
# Fitness Goal options from Gym Dataset
fitness_goal_options = ["Weight Loss", "Muscle Gain", "Weight Gain", "General Fitness"] # Defaults
if not gym_df.empty and 'Fitness_Goal' in gym_df.columns:
    unique_goals = gym_df['Fitness_Goal'].dropna().unique()
    fitness_goal_options = sorted([str(g).strip() for g in unique_goals if str(g).strip()])
    if not fitness_goal_options: # Fallback if parsing failed
        fitness_goal_options = ["Weight Loss", "Muscle Gain", "Weight Gain", "General Fitness"]
fitness_goal = st.sidebar.selectbox("Primary Fitness Goal", options=fitness_goal_options)


# Options from Diet Raw Dataset for consistency with model training
activity_options = ["Sedentary", "Moderate", "Active"]
if not diet_raw_df.empty and 'Physical_Activity_Level' in diet_raw_df.columns:
    activity_options = list(diet_raw_df['Physical_Activity_Level'].dropna().unique())
activity_level = st.sidebar.selectbox("Physical Activity Level", options=activity_options)

restriction_options = ["None", "Low_Sodium", "Low_Sugar"] # Add more if in your data
if not diet_raw_df.empty and 'Dietary_Restrictions' in diet_raw_df.columns:
    restriction_options = list(diet_raw_df['Dietary_Restrictions'].dropna().unique())
restrictions = st.sidebar.selectbox("Dietary Restrictions", options=restriction_options)

allergy_options = ["None", "Peanuts", "Gluten", "Dairy", "Shellfish"] # Common defaults
if not diet_raw_df.empty and 'Allergies' in diet_raw_df.columns:
    # Allergies might be comma-separated or need parsing. For simplicity, use unique values.
    # This part might need refinement based on how 'Allergies' is formatted in diet_raw_df
    all_allergies = set()
    for item in diet_raw_df['Allergies'].dropna().unique():
        for allergy in str(item).split(','): # Assuming comma-separated
            allergy_clean = allergy.strip()
            if allergy_clean: all_allergies.add(allergy_clean)
    if all_allergies:
        allergy_options = sorted(list(all_allergies))
        if "None" not in allergy_options: allergy_options.insert(0,"None")


selected_allergies = st.sidebar.multiselect("Known Allergies (select 'None' if no allergies)", options=allergy_options, default=["None"])
if "None" in selected_allergies and len(selected_allergies) > 1:
    selected_allergies.remove("None") # "None" is exclusive
if not selected_allergies: # If deselected all, ensure "None" is the default
    selected_allergies = ["None"]


# --- Plan Generation ---
if st.sidebar.button("✨ Generate My Plan ✨"):
    bmi = calculate_bmi(weight, height)
    bmi_cat = get_bmi_category(bmi)

    user_profile = {
        "age": age, "gender": gender, "weight": weight, "height": height, "bmi": bmi,
        "bmi_category": bmi_cat, "hypertension": hypertension, "diabetes": diabetes,
        "fitness_goal": fitness_goal, "activity_level": activity_level,
        "restrictions": restrictions, "allergies": selected_allergies,
        # For saving, you might want to add a timestamp or ID
        "timestamp": pd.Timestamp.now()
    }

    # Store profile (simulating database)
    if 'user_id' not in st.session_state: # Simple way to give a user an ID for the session
        st.session_state.user_id = f"user_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    
    profile_to_save = user_profile.copy()
    profile_to_save['user_id'] = st.session_state.user_id
    profile_to_save['allergies'] = ", ".join(selected_allergies) # Store as string for CSV
    save_user_profile(profile_to_save)


    # --- Display Results ---
    st.header(f"📜 Your Personalized Plan (BMI: {bmi} - {bmi_cat})")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏋️ Workout Plan")
        if gym_df.empty:
            st.warning("Workout data is not loaded. Cannot generate workout plan.")
        else:
            workout_rec = get_workout_recommendation(user_profile, gym_df)
            if "error" in workout_rec:
                st.warning(workout_rec["error"])
                # Try to find any plan for the goal if exact match failed
                generic_rec = gym_df[gym_df['Fitness_Goal'].str.contains(fitness_goal, case=False, na=False)]
                if not generic_rec.empty:
                    st.info("Showing a more general plan for your goal:")
                    plan = generic_rec.iloc[0]
                    st.markdown(f"**Fitness Type:** {plan.get('Fitness_Type', 'N/A')}")
                    st.markdown(f"**Suggested Exercises:** {plan.get('Exercises', 'N/A')}")
                    st.markdown(f"**Equipment:** {plan.get('Equipment', 'N/A')}")
                else:
                    st.error("No workout information found for this goal.")

            else:
                st.markdown(f"**Fitness Type:** {workout_rec.get('Fitness Type', 'N/A')}")
                st.markdown(f"**Suggested Exercises:**")
                for ex in str(workout_rec.get('Exercises', '')).split(','): # Assuming comma separated
                    st.markdown(f"- {ex.strip()}")
                st.markdown(f"**Equipment:** {workout_rec.get('Equipment', 'N/A')}")
                st.markdown(f"**General Diet Tip:** {workout_rec.get('General Diet Advice (from workout plan)', 'N/A')}")
                st.info(f"**Coach Advice:** {workout_rec.get('General Recommendation', 'N/A')}")


    with col2:
        st.subheader("🥗 Diet Recommendation")
        if diet_model_pipeline is None:
            st.warning("Diet model is not loaded. Cannot generate diet recommendation.")
        elif diet_raw_df.empty:
            st.warning("Raw diet data for options is not loaded. Cannot generate diet recommendation.")
        else:
            diet_pred = get_diet_prediction(user_profile, diet_model_pipeline, diet_raw_df)
            st.success(f"**Recommended Diet Focus: {diet_pred}**")
            
            # Add more detailed diet info based on prediction if available
            if diet_pred == "Low_Carb":
                st.markdown("""
                Focus on:
                - Lean proteins (chicken, fish, tofu)
                - Non-starchy vegetables (broccoli, spinach, peppers)
                - Healthy fats (avocado, nuts, olive oil)
                Limit: Sugary foods, refined grains (white bread, pasta), starchy vegetables in excess.
                """)
            elif diet_pred == "Low_Sodium":
                st.markdown("""
                Focus on:
                - Fresh fruits and vegetables
                - Unprocessed meats and fish
                - Whole grains
                Limit: Processed foods, canned soups, salty snacks, restaurant meals high in sodium. Check labels!
                """)
            elif diet_pred == "Balanced":
                st.markdown("""
                Focus on:
                - A variety of fruits and vegetables (all colors)
                - Whole grains (oats, brown rice, quinoa)
                - Lean proteins (beans, lentils, poultry, fish)
                - Healthy fats in moderation
                Limit: Processed foods, sugary drinks, excessive saturated and trans fats.
                """)
            else: # If other categories exist
                st.info("General healthy eating principles apply: prioritize whole foods, hydration, and portion control.")

    st.info("Remember: This is a general recommendation. Consult with a healthcare professional or certified coach before starting any new fitness or diet program.")

else:
    st.info("⬅️ Please fill in your details in the sidebar and click 'Generate My Plan'.")


# --- (Optional) Display User Profile History ---
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show My Saved Profiles", False):
    if os.path.exists(USER_PROFILES_DB):
        profiles_df = pd.read_csv(USER_PROFILES_DB)
        # Filter for current session user if ID exists, otherwise show all
        if 'user_id' in st.session_state:
            st.sidebar.dataframe(profiles_df[profiles_df['user_id'] == st.session_state.user_id].tail())
        else:
            st.sidebar.dataframe(profiles_df.tail())

    else:
        st.sidebar.info("No profiles saved yet.")