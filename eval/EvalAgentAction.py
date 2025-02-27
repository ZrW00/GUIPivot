from .utils import qwen_translate_action, ActionType
from .models import Qwen2VLAgent
import math
import torch
import json
from tqdm import tqdm
import re, os
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from rouge import Rouge
from typing import *
import copy



def androidControlActionJudge(rawPredictAction:str, rawGroundTruthAction:str, layoutPath:str, width:int=1080, height:int=2400):
    def __in_bbox(x, y, bbox):
        return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    
        
    if predictAction.action_type != groundTruthAction.action_type:
        return False
    elif predictAction.action_type == ActionType.TYPE:
        return predictAction.typed_text == groundTruthAction.typed_text
    elif predictAction.action_type == ActionType.OPENAPP:
        return predictAction.app_name == groundTruthAction.app_name
    elif predictAction.action_type == ActionType.CLICK or predictAction.action_type == ActionType.LONGCLICK:
        x1, y1 = predictAction.point
        x2, y2 = groundTruthAction.point
        x2f, y2f = x2 / 1000 * width, y2 / 1000 * height
        
        with open(layoutPath, "r") as f:
            layouts = [json.loads(line.strip()) for line in f]
        
        inboundFlag, distFlag = False, False
        
        for layout in layouts:
            if layout["is_clickable"] and "bbox_pixels" in layout and __in_bbox(x2f, y2f, layout["bbox_pixels"]):
                inboundFlag = __in_bbox(x1 / 1000 * width, y1 / 1000 * width, layout["bbox_pixels"])
                break
    
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distFlag = distance <= 140
        return inboundFlag or distFlag
    return True

def actionTypeJudge(rawPredictAction:str, rawGroundTruthAction:str):
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    return predictAction.action_type == groundTruthAction.action_type

def aitzActionJudge(rawPredictAction:str, rawGroundTruthAction:str, bboxes:list):
    def __in_bbox(x, y, bbox):
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    
    if rawGroundTruthAction.startswith("action:"):
        rawGroundTruthAction = rawGroundTruthAction.replace("action:", "")
    if rawPredictAction.startswith("action:"):
        rawPredictAction = rawPredictAction.replace("action:", "")
    predictAction = qwen_translate_action(rawPredictAction)
    groundTruthAction = qwen_translate_action(rawGroundTruthAction)
    
        
    if predictAction.action_type != groundTruthAction.action_type:
        return False
    elif predictAction.action_type == ActionType.TYPE:
        return predictAction.typed_text == groundTruthAction.typed_text
    elif predictAction.action_type == ActionType.OPENAPP:
        return predictAction.app_name == groundTruthAction.app_name
    elif predictAction.action_type == ActionType.CLICK or predictAction.action_type == ActionType.LONGCLICK:
        x1, y1 = predictAction.point
        x2, y2 = groundTruthAction.point
        
        inboundFlag, distFlag = False, False
        
        for bbox in bboxes:
            if __in_bbox(x2, y2, bbox):
                inboundFlag = __in_bbox(x1, y1, bbox)
                break
    
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distFlag = distance <= 140
        return inboundFlag or distFlag
    return True



def androidControlActionEval(
    modelPath:str, 
    testJsonPath:str,
    agentType:str="mllm",
    cache_dir='~/.cache', 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_new_tokens=128,
    log:str="seed42"
):
    def extractAction(text:str):
        action_pattern = r"action: (.+)"
        try:
            actionMatch = re.search(action_pattern, text)
            if actionMatch:
                action = actionMatch.group(1)
            else:
                action = ""
        except Exception as e:
            action = text.replace("action:", "").replace("actions:", "").strip().strip("\n")
        return action
    agent = Qwen2VLAgent(modelPath, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens)
    with open(testJsonPath, 'r') as f:
        testData = json.load(f)
    actionSuccess, typeSuccess = 0, 0
    records = []
    typeDetail = {
        "Total":{"type match": 0, "action match":0, "sample":0, "type match rate": 0, "action match rate":0},
        "SCROLL":{"type match": 0, "action match":0, "sample":0, "type match rate": 0, "action match rate":0},
        "PRESS":{"type match": 0, "action match":0, "sample":0, "type match rate": 0, "action match rate":0}
    } 
    for i, sample in tqdm(enumerate(testData), total=len(testData), desc=f"Evaluate {testJsonPath}"):
        # try:
        record = copy.deepcopy(sample)
        if agentType in ["mllm", "qwen25vl"]:
            gt = sample["messages"][1]["content"]
        else:
            gt = sample["output"]
        gtAction = extractAction(gt)
        gtActionFormated = qwen_translate_action(gtAction)
        gtActionType = ActionType(gtActionFormated.action_type).name
        
        
        pred = agent.generate(sample=sample)[0]
        try:
            predAction = extractAction(pred)
            typeFlag = int(actionTypeJudge(predAction, gtAction))
            flag = int(androidControlActionJudge(predAction, gtAction, sample["layouts"][0], sample["width"], sample["height"]))
        except Exception as e:
            print(f"Predict Action: {predAction} raises the following exception:\n {e}")
            flag = 0
            typeFlag = 0
        record["ground truth action"] = gtAction
        record['pred'] = pred
        record["pred action"] = predAction
        record["action match"] = flag
        record["action type"] = gtActionType
        record["type match"] = typeFlag
        
        if gtActionType not in typeDetail:
            typeDetail[gtActionType] = {"type match": typeFlag, "action match":flag, "sample":1}
        else:
            typeDetail[gtActionType]["type match"] += typeFlag
            typeDetail[gtActionType]["action match"] += flag
            typeDetail[gtActionType]["sample"] += 1
        typeDetail["Total"]["type match"] += typeFlag
        typeDetail["Total"]["action match"] += flag
        typeDetail["Total"]["sample"] += 1
        if gtActionType.startswith("PRESS"):
            typeDetail["PRESS"]["type match"] += typeFlag
            typeDetail["PRESS"]["action match"] += flag
            typeDetail["PRESS"]["sample"] += 1
        if gtActionType.startswith("SCROLL"):
            typeDetail["SCROLL"]["type match"] += typeFlag
            typeDetail["SCROLL"]["action match"] += flag
            typeDetail["SCROLL"]["sample"] += 1
        
        records.append(record)
        actionSuccess += flag
        typeSuccess += typeFlag
    matchRate = actionSuccess / len(testData)
    typeRate = typeSuccess / len(testData)
    with open(os.path.join('./analyses/', f'{log}.json'), "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    
    for typename, _ in typeDetail.items():
        typeDetail[typename]["type match rate"] = round(100 * typeDetail[typename]["type match"] / typeDetail[typename]["sample"], 2)
        typeDetail[typename]["action match rate"] = round(100 * typeDetail[typename]["action match"] / typeDetail[typename]["sample"], 2)

    
    print(f"Action Success Rate of {testJsonPath} is: {matchRate * 100:.2f}, Type Match Rate is: {typeRate * 100:.2f}")
    
    print(json.dumps(typeDetail, indent=4))

    return matchRate

def aitzAblationActionEval(
    modelPath:str, 
    testJsonPath:str,
    agentType:str="mllm",
    cache_dir='~/.cache', 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_new_tokens=128,
    log:str="seed42"
):
    def extractAction(text:str):
        if text.startswith("{") or text.startswith("\n{"):
            try:
                action = json.loads(text)
                action = action["Action Decision"]
            except json.JSONDecodeError:
                match = re.search(r'"Action Decision":\s*"(.*?)"', text)
                action = match.group(1) if match is not None else ""
            except Exception as e:
                action = ""
        else:
            try:
                match = re.search(r'Action Decision:\s*(.*)', text)
                action = match.group(1) if match is not None else ""
            except Exception as e:
                action = text.replace("Action Decision:", "").strip().strip("\n")
        return action
            
    agent = Qwen2VLAgent(modelPath, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens)
    with open(testJsonPath, 'r') as f:
        testData = json.load(f)
    actionSuccess, typeSuccess = 0, 0
    records = []
    typeDetail = {
        "Total":{"type match": 0, "action match":0, "sample":0, "type match rate": 0, "action match rate":0},
        "SCROLL":{"type match": 0, "action match":0, "sample":0, "type match rate": 0, "action match rate":0},
        "PRESS":{"type match": 0, "action match":0, "sample":0, "type match rate": 0, "action match rate":0}
    } 
    for i, sample in tqdm(enumerate(testData), total=len(testData), desc=f"Evaluate {testJsonPath}"):
        record = copy.deepcopy(sample)
        if agentType in ["mllm", "qwen25vl"]:
            gt = sample["messages"][1]["content"]
        else:
            gt = sample["output"]

        
        gtAction = extractAction(gt)
        
            
        gtActionFormated = qwen_translate_action(gtAction)
        gtActionType = ActionType(gtActionFormated.action_type).name
            
        pred = agent.generate(sample=sample)[0]
        try:
            predAction = extractAction(pred)
            flag = int(aitzActionJudge(predAction, gtAction, sample["bboxes"]))
            typeFlag = int(actionTypeJudge(predAction, gtAction))
        except Exception as e:
            print(f"Predict: {pred} raises the following exception:\n {e}")
            flag = 0
            typeFlag = 0
        record['pred'] = pred
        record["pred action"] = predAction
        record["action match"] = flag
        record["action type"] = gtActionType
        record["type match"] = typeFlag
        if gtActionType not in typeDetail:
            typeDetail[gtActionType] = {"type match": typeFlag, "action match":flag, "sample":1}
        else:
            typeDetail[gtActionType]["type match"] += typeFlag
            typeDetail[gtActionType]["action match"] += flag
            typeDetail[gtActionType]["sample"] += 1
        typeDetail["Total"]["type match"] += typeFlag
        typeDetail["Total"]["action match"] += flag
        typeDetail["Total"]["sample"] += 1
        if gtActionType.startswith("PRESS"):
            typeDetail["PRESS"]["type match"] += typeFlag
            typeDetail["PRESS"]["action match"] += flag
            typeDetail["PRESS"]["sample"] += 1
        if gtActionType.startswith("SCROLL"):
            typeDetail["SCROLL"]["type match"] += typeFlag
            typeDetail["SCROLL"]["action match"] += flag
            typeDetail["SCROLL"]["sample"] += 1
        records.append(record)
        actionSuccess += flag
        typeSuccess += typeFlag
    matchRate = actionSuccess / len(testData)
    typeRate = typeSuccess / len(testData)
    
    with open(os.path.join('./analyses/', f'{log}.json'), "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    
    for typename, _ in typeDetail.items():
        typeDetail[typename]["type match rate"] = round(100 * typeDetail[typename]["type match"] / typeDetail[typename]["sample"], 2)
        typeDetail[typename]["action match rate"] = round(100 * typeDetail[typename]["action match"] / typeDetail[typename]["sample"], 2)

    
    print(f"Action Success Rate of {testJsonPath} is: {matchRate * 100:.2f}, Type Match Rate is: {typeRate * 100:.2f}")
    
    print(json.dumps(typeDetail, indent=4))

    return matchRate
