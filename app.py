
import os
import json
import sys
import streamlit as st
from streamlit.components.v1 import html
from utils.data_generation import main
from utils.retrieval import retrieve_context
from utils.evaluation import main1
from utils.ranking import evaluate_prompt

# Set the page title and icon
st.set_page_config(page_title="Prompt Generation App", page_icon="🤖", layout="wide")

# --- Style the page ---
st.markdown(
    """
    <style>
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        .stTextInput {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
        }
        .stButton {
            background-color: white;
            color: black;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton:hover {
            background-color: #f2f2f2;
        }
        .stTable {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .stTable th, .stTable td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        .stTable th {
            background-color: #f2f2f2;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
st.sidebar.header("Search History")

# search_history_text_area = st.sidebar.text_area("Search History:", height=200)
# uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "json","pdf"])

# --- Main Content ---
st.title("Prompt Generation App 🤖")
# Create a list to store the search history
search_history_list = []

# Create a text input field for the user to enter their query
query_input = st.text_input("Enter your query:", value="")

# Store the last search query
if query_input:
    search_history_list.append(query_input)
    if len(search_history_list) > 5:
        search_history_list = search_history_list[-5:]  # keep only the last 5 searches

    # Create the text area with the updated search history
    st.sidebar.text_area("Search History:", height=200, value="\n".join(search_history_list))
    
# Add an upload button for files
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "json", "pdf"])


# Buttons for Generate Prompts and Evaluate Prompts
# Buttons for Generate Prompts and Evaluate Prompts
col1, col2, col3 = st.columns(3)
with col1:
    generate_button = st.button("Generate Prompts")
with col2:
    evaluate_button = st.button("Evaluate Prompts")
with col3:
    rank_button = st.button("Rank Prompts")

# Variables to store generated data
questions = []
accuracies = []
ranks = []

# --- Generate Prompts Section ---
if generate_button:
    # Retrieve context based on the user query
    context_docs = retrieve_context(query_input)
    context = " ".join([doc.page_content for doc in context_docs])

    # Call the main function from data_generator.py
    main("5", query_input)  # pass the number of test outputs as an argument

    # Load the generated data from test_data.json
    test_data_path = os.path.join("test_dataset", "test_data.json")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Display only the questions
    questions = [item["user"] for item in test_data]
    st.write("Generated Questions:")
    st.write(questions)  # Display questions directly

# --- Evaluate Prompts Section ---
if evaluate_button:
    # Load the generated data from test_data.json
    test_data_path = os.path.join("test_dataset", "test_data.json")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Display only the questions
    questions = [item["user"] for item in test_data]
    accuracies = main1()  # Call the main1 function

    # Sort the data by accuracy in descending order
    sorted_data = sorted(zip(questions, accuracies), key=lambda x: x[1], reverse=True)

    # Get the ranks
    ranks = [i + 1 for i in range(len(sorted_data))]

    # Display the sorted data
    st.write("Evaluation Results:")
    st.table({"Rank": ranks, "Questions": [x[0] for x in sorted_data], "Accuracy": [x[1] for x in sorted_data]})

if rank_button:
    # Load the generated data from test_data.json
    test_data_path = os.path.join("test_dataset", "test_data.json")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Get the questions from test_data
    questions = [item["user"] for item in test_data]

    # Evaluate the prompts using both Monte Carlo and Elo
    evaluations = evaluate_prompt(query_input, questions)

    # Extract Monte Carlo scores and Elo ratings
    monte_carlo_scores = [evaluations[f'test_case_{idx+1}']['Monte Carlo Evaluation'] 
                          for idx in range(len(questions))]
    elo_ratings = [evaluations[f'test_case_{idx+1}']['Elo Rating Evaluation'] 
                   for idx in range(len(questions))]

    # Create a table to display the rankings
    st.write("Ranked Prompts (Elo and Monte Carlo):")
    st.table(
        {
            "Prompt": questions,
            "Monte Carlo Score": monte_carlo_scores,
            "Elo Rating": elo_ratings
        }
    )