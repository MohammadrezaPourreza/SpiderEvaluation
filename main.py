import pandas as pd
import subprocess
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentExecutor

SPIDER_DATABASE_DIR = "spider"
llm = OpenAI(
    model_name="text-davinci-003",
    openai_api_key="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",
    temperature=0
)
def count_tokens(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result
def load_database(SPIDER_DATABASE_DIR):
    dev_df = pd.read_json(SPIDER_DATABASE_DIR+"/dev.json")
    dev_df = dev_df.drop(columns = ['query_toks', 'query_toks_no_value', 'question_toks','sql'], axis=1)
    schema_df = pd.read_json(SPIDER_DATABASE_DIR+"/tables.json")
    return dev_df,schema_df
def run(dev_df):
    results = []
    for index, row in dev_df.iterrows():
        if index > 99 or index < 51: continue
        print(f"Current index is {index}")
        print(row['question'])
        print(row['query'])
        db = SQLDatabase.from_uri("sqlite:///"+SPIDER_DATABASE_DIR+'/database/'+row['db_id']+"/"+row['db_id']+".sqlite")
        toolkit = SQLDatabaseToolkit(db=db,llm = llm)
        agent_executor = create_sql_agent(
            llm= llm,
            toolkit=toolkit,
            verbose=False,
        )
        print(agent_executor.agent.get_allowed_tools())
        # agent_executor.return_intermediate_steps = True
        # response = count_tokens(agent_executor,row['question'])
        # for agent_action in response["intermediate_steps"]:
        #     if agent_action[0][0] == 'query_sql_db':
        #         Predicted_SQL = agent_action[0][1]
        #     else:
        #         Predicted_SQL = "NO SQL detected"
        # print(Predicted_SQL)
        # print("=================================")
        # results.append([row['question'], Predicted_SQL, row['query'], row['db_id']])
    return results
def evaluation(results_dir):
    command = "cp " + results_dir + " test-suite-sql-eval-master/results"
    subprocess.run(command, shell=True, capture_output=False, text=True)
    command = "python3 test-suite-sql-eval-master/OutputProcessing.py"
    subprocess.run(command, shell=True, capture_output=False, text=True)
    command = "rm test-suite-sql-eval-master/results/test.csv"
    subprocess.run(command, shell=True, capture_output=False, text=True)
    cmd_str = "python3 test-suite-sql-eval-master/evaluation.py --gold test-suite-sql-eval-master/processed_results/Gold_test.txt --pred test-suite-sql-eval-master/processed_results/Predicted_test.txt --db test-suite-sql-eval-master/database/ --etype exec "
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    command = "rm test-suite-sql-eval-master/processed_results/Gold_test.txt"
    subprocess.run(command, shell=True, capture_output=False, text=True)
    command = "rm test-suite-sql-eval-master/processed_results/Predicted_test.txt"
    subprocess.run(command, shell=True, capture_output=False, text=True)
    return float(result.stdout[-21:-16])

if __name__ == '__main__':
    dev_df,schema_df = load_database(SPIDER_DATABASE_DIR)
    results = run(dev_df)
    # df = pd.DataFrame(results, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
    # df.to_csv("test.csv", index=False)
    # results = evaluation("test.csv")
    # print(f"The execution accuracy is {results}")