import streamlit as st
from utils.constants import *
import torch
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

st.title("ኬፍ talk to my AI Assistant")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles_chat.css")

# Get the variables from constants.py
pronoun = info['Pronoun']
name = info['Name']
subject = info['Subject']
full_name = info['Full_Name']

# Initialize the chat history
if "messages" not in st.session_state:
    welcome_msg = f"Hi! I'm {name}'s AI Assistant, Buddy. How may I assist you today?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# App sidebar
with st.sidebar:
    st.markdown("""
                # Chat with my AI assistant
                """)
    with st.expander("Click here to see FAQs"):
        st.info(
            f"""
            - What are {pronoun} strengths and weaknesses?
            - What is {pronoun} expected salary?
            - What is {pronoun} latest project?
            - When can {subject} start to work?
            - Tell me about {pronoun} professional background
            - What is {pronoun} skillset?
            - What is {pronoun} contact?
            - What are {pronoun} achievements?
            """
        )
    
    import json
    messages = st.session_state.messages
    if messages is not None:
        st.download_button(
            label="Download Chat",
            data=json.dumps(messages),
            file_name='chat.json',
            mime='json',
        )

    st.caption(f"Â© Made by {full_name} 2023. All rights reserved.")


with st.spinner("Initiating the AI assistant. Please hold..."):
    # Check for GPU availability and set the appropriate device for computation.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Global variables
    llm_hub = None
    embeddings = None
    
    Watsonx_API = "HaAse7ogOtxHmx-eFIZPREvr_cczgSsQO0sIW4Ny3B-H"
    Project_id = "f86a3aa8-31a8-43d6-be95-c1ab12df935c"

    # Function to initialize the language model and its embeddings

    def init_watson_assistant():
        global assistant
    
        authenticator = IAMAuthenticator('HaAse7ogOtxHmx-eFIZPREvr_cczgSsQO0sIW4Ny3B-H')
        assistant = AssistantV2(
            version='2023-11-23',
            authenticator=authenticator
        )
        assistant.set_service_url('https://api.au-syd.assistant.watson.cloud.ibm.com/instances/2c797ed4-bf50-4a03-bf26-8c8031cf9e55')
        assistant_id = 'ab4f6ec4-3ca4-4471-8fc9-5f28fbd30828'
    init_watson_assistant()
    def ask_watson_assistant(user_query):
        global assistant
    
        message_input = {
            'message_type': 'text',
            'text': user_query
        }
    
        result = assistant.message_stateless(
            assistant_id,
            input=message_input
        ).get_result()

        # Extract and return text responses
        responses = [response['text'] for response in result.get('output', {}).get('generic', []) if response['response_type'] == 'text']
        return '\n'.join(responses)

# ... (remaining code remains unchanged)

def ask_bot(user_query):
    # Use the Watson Assistant code to get responses
    return ask_watson_assistant(user_query)
  
    # load the file
    documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()
    
    # LLMPredictor: to generate the text response (Completion)
    llm_predictor = LLMPredictor(
            llm=llm_hub
    )
                                    
    # Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
    embed_model = LangchainEmbedding(embeddings)
    
    # ServiceContext: to encapsulate the resources used to create indexes and run queries    
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            embed_model=embed_model
    )      
    # build index
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

def ask_bot(user_query):

    global index

    PROMPT_QUESTION = """You are Buddy, an AI assistant dedicated to assisting {name} in {pronoun} job search by providing recruiters with relevant information about {pronoun} qualifications and achievements. 
    Your goal is to support {name} in presenting {pronoun}self effectively to potential employers and promoting {pronoun} candidacy for job opportunities.
    If you do not know the answer, politely admit it and let recruiters know how to contact {name} to get more information directly from {pronoun}. 
    Don't put "Buddy" or a breakline in the front of your answer.
    Human: {input}
    """
    
    # query LlamaIndex and LLAMA_2_70B_CHAT for the AI's response
    output = index.as_query_engine().query(PROMPT_QUESTION.format(name=name, pronoun=pronoun, input=user_query))
    return output

# After the user enters a message, append that message to the message history
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Iterate through the message history and display each message
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("ðŸ¤” Thinking..."):
            response = ask_bot(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

# Suggested questions
questions = [
    f'What are {pronoun} strengths and weaknesses?',
    f'What is {pronoun} latest project?',
    f'When can {subject} start to work?'
]

def send_button_ques(question):
    st.session_state.disabled = True
    response = ask_bot(question)
    st.session_state.messages.append({"role": "user", "content": question}) # display the user's message first
    st.session_state.messages.append({"role": "assistant", "content": response.response}) # display the AI message afterwards
    
if 'button_question' not in st.session_state:
    st.session_state['button_question'] = ""
if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False
    
if st.session_state['disabled']==False: 
    for n, msg in enumerate(st.session_state.messages):
        # Render suggested question buttons
        buttons = st.container()
        if n == 0:
            for q in questions:
                button_ques = buttons.button(label=q, on_click=send_button_ques, args=[q], disabled=st.session_state.disabled)
