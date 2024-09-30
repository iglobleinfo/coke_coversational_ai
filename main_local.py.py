import mammoth

from langchain_openai import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from langchain.vectorstores import Chroma



import pandas as pd
from datetime import timedelta
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from flask import Flask, request, jsonify

app = Flask(__name__)


chat_history = []  # Initialize chat history
# last_vin=[]
OPENAI_API_KEY= "OPEN AI API KEY"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)




# Load CSV files
file1 = "C:/Users/saksh/OneDrive/Documents/BytEdge/coke/Data analysis tool/Data/ccswb-workorders-parts-090324.csv"
file2 = "C:/Users/saksh/OneDrive/Documents/BytEdge/coke/Data Analysis coke/Data/processed_ccswb-faults-090224.csv"
file3="C:/Users/saksh/OneDrive/Documents/BytEdge/coke/Data Analysis coke/Data/data.csv"
file4="C:/Users/saksh/OneDrive/Documents/BytEdge/coke/Data Analysis coke/Data/coke.csv"



df1 = pd.read_csv(file1, encoding='utf-8', on_bad_lines='skip')
df2 = pd.read_csv(file2, encoding='utf-8', on_bad_lines='skip')
df3 = pd.read_csv(file3, encoding='utf-8', on_bad_lines='skip')
df4 = pd.read_csv(file4, encoding='utf-8', on_bad_lines='skip')

# Convert PartsQuantity and PartsAmount to positive values
df3['PartsQuantity'] = df3['PartsQuantity'].abs()
df3['PartsAmount'] = df3['PartsAmount'].abs()

df3['WorkOrderStartDate'] = pd.to_datetime(df3['WorkOrderStartDate'], errors='coerce')
df3['WorkOrderFinishDate'] = pd.to_datetime(df3['WorkOrderFinishDate'], errors='coerce')

############ LLM QUERY MODELS ###########################
def query_model(prompt):
    response = llm(prompt)
    return response

def generate_response_summary(vin_summary):
    prompt = f"Summarize the following vin data:\n\n{vin_summary}"
    response = query_model(prompt)
    return response


def generate_response_predict(next_maintenance_date):
    prompt = f"Based on the following maintenance intervals for VIN next maintenance date predicted is :{next_maintenance_date}"
    print(prompt)

    response = query_model(prompt)
    print(response)
    return response

################ AGENTS FOR DATA ANALYSIS ####################################

# Create an agent for dataframe analysis
agent_executor1 = create_pandas_dataframe_agent(
    llm,
    df=df1,
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True
)

agent_executor2 = create_pandas_dataframe_agent(
    llm,
    df=df2,
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True
)

agent_executor3 = create_pandas_dataframe_agent(
    llm,
    df=df4,
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True
)

##### USECASE 1: Function to summarize VIN-related information #####################
def get_vin_summary(vin_number):
    # Assuming df3 is for parts and maintenance data and df2 is for fault data
    df_vin = df3[df3['vin'].astype(str) == str(vin_number)]  # Dataframe for parts and maintenance data
    df_faults = df2[df2['Vin'].astype(str) == str(vin_number)]  # Dataframe for fault data

    # Check if there is any data for the given VIN
    if df_vin.empty and df_faults.empty:
        return "No data found for the given VIN."

    # Summarizing parts and maintenance data
    parts_changed = df_vin['PartsDescription'].unique().tolist() if not df_vin.empty else []
    parts_count = df_vin['PartsDescription'].value_counts().to_dict() if not df_vin.empty else {}
    asset_plan_loc = df_vin['AssetPlantLocation'].unique().tolist() if not df_vin.empty else []
    asset_manufacturer = df_vin['AssetManufacturer'].unique().tolist() if not df_vin.empty else []
    asset_description = df_vin['AssetDescription'].unique().tolist() if not df_vin.empty else []

    # Summarizing fault data
    fault_codes = df_faults['Fault Code'].unique().tolist() if not df_faults.empty else []
    fault_code_counts = df_faults['Fault Code'].value_counts().to_dict() if not df_faults.empty else {}
    failure_codes = df_faults['Failure Code'].unique().tolist() if not df_faults.empty else []
    failure_code_counts = df_faults['Failure Code'].value_counts().to_dict() if not df_faults.empty else {}

    # if not df_faults.empty:
    #     # Custom date parsing for the given format
    #     def custom_date_parser(date_str):
    #         try:
    #             return pd.to_datetime(date_str, format='%a %b %d %Y %H:%M:%S %Z%z', errors='coerce')
    #         except Exception as e:
    #             print(f"Error parsing date: {e}")
    #             return pd.NaT
        
    #     # Parse 'Sent Date' with the custom parser
    #     df_faults['Sent Date'] = df_faults['Sent Date'].apply(custom_date_parser)

    # Sort by 'Sent Date' to get the most recent entries
    recent_faults = df_faults.sort_values(by='Sent Date', ascending=False).head(5)
    # Extract relevant information: 'Sent Date', 'Fault Code', 'Failure Code'
    recent_faults_summary = recent_faults[['Sent Date', 'Fault Code', 'Failure Code']].to_dict('records')
    
    print(recent_faults_summary)
    
    # Create a summary dictionary
    summary = {
        "Asset Plant Location": asset_plan_loc,
        "Manufacturer": asset_manufacturer,
        "Asset Description": asset_description,
        "Parts Changed": parts_changed,
        "Parts Count": parts_count,
        "Fault Codes": fault_codes,
        "Fault Code Counts": fault_code_counts,
        "Failure Codes": failure_codes,
        "Failure Code Counts": failure_code_counts,
        "Most Recent Faults and Failures": recent_faults_summary
    }

    return summary

#### USECASE 2: Predict next maintenance date of the VIN number.

def predict_next_maintenance(vin):
    print("maintenance",vin)
    # Filter data for a specific VIN and sort by date
    data_new=df3[(df3['vin']==vin) & (df3['MaintenanceActivityTypeDescription']=='Preventive Maintenance')]
    
    print(data_new)
    if data_new.empty:
        return "No preventive maintenance data found for this VIN number."
    df_vin = data_new.sort_values(by='WorkOrderFinishDate')
    
    print(df_vin)
    # Calculate intervals between maintenance activities
    df_vin['PreviousDate'] = df_vin['WorkOrderFinishDate'].shift(1)
    df_vin['IntervalDays'] = (df_vin['WorkOrderFinishDate'] - df_vin['PreviousDate']).dt.days

    # Drop rows where intervals are 0 days
    df_vin = df_vin[df_vin['IntervalDays'] != 0]
    # Calculate average interval excluding 0 day intervals
    average_interval = df_vin['IntervalDays'].mean()
    last_maintenance_date = df_vin['WorkOrderFinishDate'].iloc[-1]
    predicted_next_maintenance = last_maintenance_date + timedelta(days=int(average_interval))
    print(predicted_next_maintenance)
    # Format the date to exclude time
    formatted_date = predicted_next_maintenance.strftime('%Y-%m-%d')
    date_predicted = {
        "next_predicted_date":formatted_date
        
    }
    return date_predicted
    
######## USECASE 3: ############################ AVERAGE VALUE #########################
def calculate_part_requirements(part_number, location, months=1):
    # Filter data for the specified part number and location
    relevant_data = df3[(df3['PartNumber'] == part_number) & (df3['AssetPlantLocation'].str.contains(location))]
    
    # Extract month and year from the date column (assuming there is a date column named 'Date')
    relevant_data['YearMonth'] = pd.to_datetime(relevant_data['WorkOrderStartDate']).dt.to_period('M')

    # Group by 'YearMonth' and calculate the total sum of 'PartsQuantity' for each month
    monthly_data = relevant_data.groupby('YearMonth')['PartsQuantity'].sum().reset_index()

    print(monthly_data)

    # Calculate the average monthly parts quantity
    average_monthly_quantity = int(monthly_data['PartsQuantity'].mean())
    # Calculate the required parts for the next month (or the specified number of months)
    required_quantity = average_monthly_quantity * months
    
    required = {
        'total_parts_required_in_next_month_for_engine_oil': required_quantity
    }
    
    return required

######## USECASE 4: ############################ Vehicle health #########################

# def predict_health(vin):
#     print(vin)
#     # Filter data for a specific VIN and sort by date
#     data_new=df4[df4['vin']==vin]
    
    
#     return date_predicted

########################### USE CASE 5: Unstructured document ##########################################

# loader = Docx2txtLoader("/home/ubuntu/coke_chat/Data Analysis coke/Data/protocol ,fault code, failure code.docx")
loader=Docx2txtLoader("C:/Users/saksh/OneDrive/Documents/BytEdge/coke/Data Analysis coke/Data/protocol ,fault code, failure code.docx")
documents=loader.load()
# Split the document text into chunks for embeddings
document_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=100)
document_chunks = document_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./vectorstore2')
vectordb.persist()
# Create a conversational retrieval QA chain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                   retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                                                   verbose=False,memory=memory)

####################### HANDLE QUERY ############################

def handle_user_query(user_input, chat_history):
    # Extract last VIN if available
    chat_history.append({"role": "user", "content": user_input})
    # Contextualize the query with previous interactions
    context = "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in chat_history])
    print("context:",context)
    last_vin = None
    for entry in chat_history:
        if 'vin' in entry:
            last_vin = entry['vin']

    # Check if the input is about a VIN
    if user_input.lower().startswith("vin"):
        parts = user_input.split()
        if len(parts) > 1:
            last_vin = parts[1]
            chat_history.append({"role": "user", "content": user_input, "vin": last_vin})  # Store VIN in chat history
            vin_summary = get_vin_summary(last_vin)
            if isinstance(vin_summary, str):
                response = vin_summary  # Handling case where summary is a string
            else:
                response_1 = generate_response_summary(vin_summary)
                response=response_1.content

        else:
            response = "Please provide a valid VIN number after 'VIN'."
    elif 'next maintenance' in user_input.lower(): 
        if last_vin:
            next_maintenance_date = predict_next_maintenance(last_vin)
            response_1= generate_response_predict(next_maintenance_date)
            response=response_1.content

        else:
            response = "No VIN found for predicting next maintenance. Please provide a VIN."

    elif 'quarts'in user_input.lower() or 'next month' in user_input.lower():
        location = "Austin"
        oil_type = "SAE 15W-40"
        partNumber="105816"
        details=calculate_part_requirements(partNumber, location, months=1)
        response_1 = generate_response_summary(details)
        response=response_1.content

    elif "faults" in user_input.lower() or "parts" in user_input.lower() or "cost" in user_input.lower() or  "repair" in user_input.lower() or  "manufacturer" in user_input.lower() or "failure" in user_input.lower() or  "maintenance" in user_input.lower() or "workorder" in user_input.lower() or "belongs to the fault code" in user_input.lower():
        # DataFrame query processing
        response = agent_executor1.run(context) if "parts" in user_input.lower() or "manufacturer" in user_input.lower() or "repair" in user_input.lower() or  "maintenance" in user_input.lower() or "workorder" in user_input.lower() or "fault" in user_input.lower() else agent_executor2.run(context)

    elif "understanding" in user_input.lower() or "what do we mean by this fault code" in user_input.lower() or "cause" in user_input.lower() or "fix" in user_input.lower():
        # Document query processing with QA chain
        response = qa_chain({"question": context})['answer']   
    
    elif "health" in user_input.lower():
        # Document query processing with QA chain
        response = agent_executor3.run(context)

    else:
        response = "Please start your query with 'VIN' or ask about maintenance."

    
    
    # Update chat history with the user's query and assistant's response
    
    chat_history.append({"role": "assistant", "content": response})

    print(chat_history)
    return response




@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if user_input.lower() == 'exit':
        return jsonify({"response": "Goodbye!"})

    response = handle_user_query(user_input, chat_history)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run()

