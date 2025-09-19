#!/usr/bin/env python
import sys
import warnings
from .crew import build_langgraph_workflow

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# workflow locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the LangGraph workflow.
    """
    workflow = build_langgraph_workflow()
    # Example input
    inputs = {
        'query': 'What is the purpose of PDPA?'
    }
    result = workflow.invoke(inputs)
    print("LangGraph workflow result:", result)

def train():
    """
    Train the workflow for a given number of iterations.
    """
    pass

def replay():
    """
    Replay the workflow execution from a specific task.
    """
    pass

def test():
    """
    Test the workflow execution and returns the results.
    """
    pass
