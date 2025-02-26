import os

import argparse


from eval import  androidControlActionEval, aitzAblationActionEval


def eval(type:str, modelPath:str, testJsonPath:str, log:str, agentType = "mllm"):
    mapping = {
        "acactioneval":androidControlActionEval,
        "aitzablationactioneval":aitzAblationActionEval
    }
    return mapping[type](modelPath=modelPath, testJsonPath=testJsonPath, log=log, agentType=agentType)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="actioneval")
    parser.add_argument("--modelPath", type=str)
    parser.add_argument("--testJsonPath", type=str)
    parser.add_argument("--log", type=str)
    parser.add_argument("--agentType", type=str, default="mllm")
    args = parser.parse_args()
    
    eval(type=args.type, modelPath=args.modelPath, testJsonPath=args.testJsonPath, log=args.log, agentType=args.agentType)