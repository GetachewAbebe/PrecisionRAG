import os
import json
import sys

current_directory = os.getcwd()
print(current_directory)
# sys.path.insert(0, '/home/getachew/Documents/10_Academy/Week_7/PrecisionRAG')
from utils.data_generator import main
from utils.data_retriever import retrieve_context
from utils.evaluation import evaluate

import streamlit as st

# Set the page title and icon
st.set_page_config(page_title="Prompt Generation App", layout="wide")

# Create a sidebar for search history
st.sidebar.header("Search History")
search_history = st.sidebar.text_area("Search History:", height=200)

# Create a text input field for the user to enter their query
query_input = st.text_input("Enter your query:", value="")

# Create a button to generate the test data
generate_button = st.button("Generate Prompts")

if generate_button:
    # Retrieve context based on the user query
    context_docs = retrieve_context(query_input)
    context = " ".join([doc.page_content for doc in context_docs])

    # # Save the context to a file
    # with open("context.txt", "w") as f:
    #     f.write(context)

    # Call the main function from data_generator.py
    main("3")  # pass the number of test outputs as an argument

    # Load the generated data from test_data.json
    test_data_path = os.path.join("test_dataset", "test_data.json")
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Display only the questions
    questions = [item["user"] for item in test_data]

    st.write("Generated Questions:")
    for i, question in enumerate(questions, start=1):
        st.write(f"**Question {i}**")
        st.info(question)
        # accuracy = evaluate("prompt", question, context)  # Call the evaluate function
        # # st.write(f"**Accuracy: {accuracy['accuracy']}**")  # Display the accuracy