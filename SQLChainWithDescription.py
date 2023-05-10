from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional
from pydantic import Extra, Field, root_validator
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.prompts.base import BasePromptTemplate
from langchain.sql_database import SQLDatabase


class SQLChainWithDescription(Chain):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SQLDatabaseChain.from_llm(OpenAI(), db)
    """

    llm_chain: LLMChain
    description_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    database: SQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    SQL_prompt: Optional[BasePromptTemplate] = None
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    top_k: int = 5
    """Number of results to return from the query"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an SQLDatabaseChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                database = values["database"]
                prompt = values.get("prompt") or SQL_PROMPTS.get(
                    database.dialect, PROMPT
                )
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        table_description = self.description_chain.run(table_info)
        _run_manager.on_text("Tables description:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            table_description, color="yellow", verbose=self.verbose
        )
        table_info = table_info + "\n" + table_description
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)
        llm_inputs = {
            "input": input_text,
            "top_k": self.top_k,
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps = []
        sql_cmd = self.llm_chain.predict(
            callbacks=_run_manager.get_child(), **llm_inputs
        )
        intermediate_steps.append(table_description)
        intermediate_steps.append(sql_cmd)
        _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
        result = self.database.run(sql_cmd)
        intermediate_steps.append(result)
        _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
        _run_manager.on_text(result, color="yellow", verbose=self.verbose)
        # If return direct, we just set the final result equal to the sql query
        if self.return_direct:
            final_result = result
        else:
            _run_manager.on_text("\nAnswer:", verbose=self.verbose)
            input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
            llm_inputs["input"] = input_text
            final_result = self.llm_chain.predict(
                callbacks=_run_manager.get_child(), **llm_inputs
            )
            _run_manager.on_text(final_result, color="green", verbose=self.verbose)
        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result["intermediate_steps"] = intermediate_steps
        return chain_result

    @property
    def _chain_type(self) -> str:
        return "sql_database_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        db: SQLDatabase,
        description_prompt: BasePromptTemplate,
        SQL_prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> SQLChainWithDescription:
        SQL_prompt = SQL_prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=SQL_prompt)
        description_chain = LLMChain(llm=llm, prompt=description_prompt,)
        return cls(llm_chain=llm_chain,description_chain=description_chain, database=db, **kwargs)