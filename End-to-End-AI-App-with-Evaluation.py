#%% md
# ```bash
# !pip install panel jupyter_bokeh
# ```
#%% md
# # set working directory
#%%
import json
import os, sys
import jupyter_bokeh  #check package existence

rootpath = r"D:\MyDrive2\pythonprojects\class\GENAI\End-to-end-app1"
if rootpath not in sys.path:
    sys.path.append(rootpath)
os.chdir(rootpath)

envfilename = "endtoend.env"

#%%
import openai
import gradio as gr
import datetime

# Utility package for English Prompts
import utils
# predefined variables
import macros

import panel as pn  # For graphical interface

pn.extension()

context = []  #archive user query
#%% md
# # load environment variable
#%%
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(envfilename)

api_key = os.getenv("OPENAI_API_KEY")  # 'ollama'  
model = "gpt-4o-mini"  # "gpt-4o-mini"
base_url = None

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


def get_completion_from_messages(messages,
                                 model="gpt-4o-mini",
                                 temperature=0,
                                 max_tokens=500):
    '''
    Encapsulate a function to access LLM

    Parameters: 
    messages: This is a list of messages, each message is a dictionary containing role and content. The role can be 'system', 'user' or 'assistant', and the content is the message of the role.
    model: The model to be called, default is gpt-4o-mini (ChatGPT) 
    temperature: This determines the randomness of the model output, default is 0, meaning the output will be very deterministic. Increasing temperature will make the output more random.
    max_tokens: This determines the maximum number of tokens in the model output.
    '''
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,  # This determines the randomness of the model's output
        max_tokens=max_tokens,  # This determines the maximum number of tokens in the model's output
    )

    return response.choices[0].message.content
#%% md
# # method to process user message
#%%
def process_user_message(user_input, all_messages, debug=True):
    """
    Preprocess user messages
    
    Parameters:
    user_input : User input
    all_messages : Historical messages
    debug : Whether to enable DEBUG mode, enabled by default
    """
    # Delimiter
    delimiter = "```"

    # Step 1: Use OpenAI's Moderation API to check if the user input is compliant or an injected Prompt
    response = client.moderations.create(input=user_input)
    moderation_output = response.results[0]

    # The input is non-compliant after Moderation API check
    if moderation_output.flagged:
        print("Step 1: Input rejected by Moderation")
        return "Sorry, your request is non-compliant"

    # If DEBUG mode is enabled, print real-time progress
    if debug: print("Step 1: Input passed Moderation check")

    # Step 2: Extract products and corresponding categories 
    category_and_product_response = utils.find_category_and_product_only(
        user_input, utils.get_products_and_category())
    #print(category_and_product_response)
    # Convert the extracted string to a list
    category_and_product_list = utils.read_string_to_list(category_and_product_response)
    #print(category_and_product_list)

    if debug: print("Step 2: Extracted product list")

    # Step 3: Find corresponding product information
    product_information = utils.generate_output_string(category_and_product_list)
    if debug: print("Step 3: Found information for extracted products")

    # Step 4: Generate answer based on information
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers. \
    Make sure to ask the user relevant follow-up questions.
    """
    # Insert message
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]
    # Get GPT3.5's answer
    # Implement multi-turn dialogue by appending all_messages
    final_response = get_completion_from_messages(all_messages + messages)
    if debug: print("Step 4: Generated user answer")
    # Add this round of information to historical messages
    all_messages = all_messages + messages[1:]

    # Step 5: Check if the output is compliant based on Moderation API
    response = client.moderations.create(input=final_response)
    moderation_output = response.results[0]

    # Output is non-compliant
    if moderation_output.flagged:
        if debug: print("Step 5: Output rejected by Moderation")
        return "Sorry, we cannot provide that information"

    if debug: print("Step 5: Output passed Moderation check")

    # Step 6: Model checks if the user's question is well answered
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question?
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    # Request model to evaluate the answer
    evaluation_response = get_completion_from_messages(messages)
    if debug: print("Step 6: Model evaluated the answer")

    # Step 7: If evaluated as Y, output the answer; if evaluated as N, feedback that the answer will be manually corrected
    if "Y" in evaluation_response:  # Use 'in' to avoid the model possibly generating Yes
        if debug: print("Step 7: Model approved the answer.")
        return final_response, all_messages
    else:
        if debug: print("Step 7: Model disapproved the answer.")
        neg_str = "I apologize, but I cannot provide the information you need. I will transfer you to a human customer service representative for further assistance."
        return neg_str, all_messages

# user_input = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"
# response, _ = process_user_message(user_input, [])
# print(response)

#%% md
# # dump log
#%%

def loguserqueries():
    global context
    with open(macros.userlogfile, "a") as f:
        f.write("========" + str(datetime.datetime.now()) + "========" + "\n")
        f.write(str(context))
        f.write("\n")
    context = []
    return
#%% md
# # chat function to be called by graio
#%%
def collect_messages_en(user_input: str, debug=False):
    """
    Used to collect user input and generate assistant responses

    Parameters:
    debug: Used to decide whether to enable debug mode
    """
    if debug: print(f"User Input = {user_input}")
    if user_input == "":
        return
    global context
    # Call process_user_message function
    #response, context = process_user_message(user_input, context, utils.get_products_and_category(),debug=True)
    response, context = process_user_message(user_input, context, debug=False)
    context.append({'role': 'assistant', 'content': f"{response}"})
    # panels.append(
    #     pn.Row('User:', pn.pane.Markdown(user_input, width=600)))
    # panels.append(
    #     pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))
    # 
    # return pn.Column(*panels) # Contains all dialogue information

    return response

# context = []
# user_message = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"
# collect_messages_en(user_message)
#%% md
# # gradio interface setup
#%%

context = []


def greet(name):
    # return f"Hello, {name}!"
    if name == 'bye':
        loguserqueries()
        return "Goodbye!"
    return collect_messages_en(name)


demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()
#%%

#%%
