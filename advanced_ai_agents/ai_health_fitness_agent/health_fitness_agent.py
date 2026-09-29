import streamlit as st
from dotenv import load_dotenv



from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()


st.set_page_config(
    page_title="AI Health and Fitness Planner",
    layout="wide"
)

st.title("AI Health and Fitness Planner")

diet_agent=Agent(
    name="Diet Agent",
    model="groq:openai/gpt-oss-20b",
    instructions=[
        "You are a general nutrition planning assistant.",
        "Create practical and balanced meal suggestions.",
        "Consider the user's dietary preference and fitness goal.",
        "Include breakfast, lunch, dinner, and snack suggestions.",
        "Mention hydration and general nutrition considerations.",
        "Do not diagnose medical conditions.",
        "Do not prescribe medical diets or medications.",
        "Do not invent medical information.",
        "Clearly state that recommendations are general guidance.",

    ],
    markdown=True,

)

fitness_agent = Agent(
    name="Fitness Agent",
    model="groq:openai/gpt-oss-20b",
    instructions=[
        "You are a general fitness planning assistant.",
        "Create a practical workout plan based on the user's goals.",
        "Include warm-up, main workout, and cool-down.",
        "Consider the user's activity level and workout preference.",
        "Keep recommendations appropriate for a general healthy adult.",
        "Do not diagnose injuries or medical conditions.",
        "Do not provide medical treatment.",
        "Encourage professional guidance when appropriate.",
        "Do not invent information.",
    ],
    markdown=True,
)
st.subheader("👤 Your Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=100,
        value=22,
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=300.0,
        value=70.0,
    )

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=175.0,
    )

    activity = st.selectbox(
        "Activity Level",
        [
            "Sedentary",
            "Lightly Active",
            "Moderately Active",
            "Very Active",
        ],
    )


with col2:

    goal = st.selectbox(
        "Fitness Goal",
        [
            "General Fitness",
            "Weight Management",
            "Muscle Gain",
            "Improve Strength",
            "Improve Endurance",
        ],
    )

    diet = st.selectbox(
        "Dietary Preference",
        [
            "No Specific Preference",
            "Vegetarian",
            "Vegan",
            "Low Carb",
            "High Protein",
        ],
    )

    workout = st.selectbox(
        "Workout Preference",
        [
            "Gym",
            "Home Workout",
            "Running",
            "Bodyweight",
            "Mixed",
        ],
    )


# --------------------------------------------------
# Generate Plan
# --------------------------------------------------

if st.button("🚀 Generate My Plan", type="primary"):

    user_profile = f"""
    User Profile:

    Age: {age}
    Weight: {weight} kg
    Height: {height} cm
    Activity Level: {activity}
    Fitness Goal: {goal}
    Dietary Preference: {diet}
    Workout Preference: {workout}
    """

    # ----------------------------------------------
    # Diet Agent
    # ----------------------------------------------

    with st.spinner("🥗 Creating nutrition plan..."):

        diet_response = diet_agent.run(
            f"""
            Create a general nutrition plan based on this profile:

            {user_profile}

            Include:

            ## Nutrition Approach

            ## Breakfast

            ## Lunch

            ## Dinner

            ## Snacks

            ## Hydration & General Tips

            Keep the recommendations practical and concise.
            """
        )


    # ----------------------------------------------
    # Fitness Agent
    # ----------------------------------------------

    with st.spinner("🏋️ Creating fitness plan..."):

        fitness_response = fitness_agent.run(
            f"""
            Create a general fitness plan based on this profile:

            {user_profile}

            Include:

            ## Workout Approach

            ## Warm-up

            ## Main Workout

            ## Cool-down

            ## Progression Tips

            Keep the recommendations practical and concise.
            """
        )


    # ----------------------------------------------
    # Display Results
    # ----------------------------------------------

    st.success("Your plan has been generated!")

    diet_tab, fitness_tab = st.tabs(
        ["🥗 Nutrition Plan", "🏋️ Fitness Plan"]
    )

    with diet_tab:

        st.markdown(diet_response.content)


    with fitness_tab:

        st.markdown(fitness_response.content)