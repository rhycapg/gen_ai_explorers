#pip install langchain openai chromadb tiktoken tabulate sqlalchemy sqlalchemy-bigquery google-cloud-bigquery
# Import Libraries

from flask import Flask, render_template, request, redirect, jsonify

from langchain.agents.agent_types import AgentType
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain import SQLDatabaseChain,SQLDatabase
from langchain.chains import SQLDatabaseSequentialChain
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain

from sqlalchemy import *
from sqlalchemy.schema import *
from sqlalchemy.engine import create_engine

from google.cloud import bigquery
from google.api_core.exceptions import InvalidArgument

import os
import pprint


app = Flask(__name__)
chat_history = []

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('signin.html')

@app.route('/index.html',methods=['GET', 'POST'])
def index():
    # print(request.get_json()['project'])
    # if request.method == 'POST':
    #     project_fromsignin = request.get_json()['project']
    #     dataset_fromsignin = request.get_json()['dataset']
    #     return redirect(url_for('index.html'),project_temp=project_fromsignin ,dataset_temp=dataset_fromsignin)
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        print(request.form['user_input'])

        # Bigquery credentials 
        service_account_file = "creds.json" # Change to where your service account key file is located
        
        project = "gen-genai-explorers-team"
        dataset = "vehicle_database" 
        sqlalchemy_url= f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'
        
        llm = VertexAI(model_name="text-bison@001")
        db_conn = SQLDatabase.from_uri(sqlalchemy_url)
        print('db connected')

        chat_history.append(('user', user_input))   
        
        # palm api
        _DEFAULT_TEMPLATE = """You are a BigQuery expert. Given an input question, first create a syntactically correct BigQuery query to run, then look at the results of the query and return the answer to the input question.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most top 10 results using the LIMIT clause as per BigQuery. You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

        If asked for a summary on a table, run the information_schema command on the requested table.

        If there is any database error like this then return insufficient privleges.

        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        Only use the following tables:
        {table_info}

        Question: {input}"""
        # Result: "Result of the query"
        PROMPT = PromptTemplate(
            input_variables=["input", "table_info"], template=_DEFAULT_TEMPLATE
        )
        db_chain = SQLDatabaseChain.from_llm(llm, db_conn, prompt=PROMPT, verbose=True)

        try:
            output = db_chain.run(user_input)
        except Exception as e:
            print(e)
            output = "Sorry, I can't help you."
        

        # toolkit = SQLDatabaseToolkit(db=db_conn, llm=llm)
        
        # agent_executor = create_sql_agent(
        # llm=llm,
        # toolkit=toolkit,
        # verbose=True,
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        # )

        # output = agent_executor.run(user_input)




        


        # output=db_chain.run(PROMPT.format(inputz = user_input))
        print(output)
        chat_history.append(('bot', output ))
        
        # Call your chatbot logic here to generate the output based on user input
        # Replace the example logic below with your own chatbot implementation
        
        user_input = "Input: " + user_input
        output = "Output: " + output
        
        return render_template('index.html', chat_history=chat_history , project_temp=project)
    return render_template('index.html',project_temp=project)
# , dataset_temp=dataset_fromsignin
# , dataset_temp=dataset
if __name__ == '__main__':
    app.run(debug=True)
