import os, json, re, copy, math
from templates import AITZACTIONPREDICTSAMPLE, AITZACTIONPREDICTPROMPTPart1
import random
from collections import Counter
from tqdm import tqdm
from PIL import Image
from string import Template


AITZACTIONPREDICTPROMPTPart2_1_5_6_7 = """
### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
- **Screenshot**: <image>
"""

AITZACTIONPREDICTPROMPTPart2_2_8 = """
### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
- **Current Screen Description**: {SD}
- **Screenshot**: <image>
"""

AITZACTIONPREDICTPROMPTPart2_3_9 = """
### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
- **Previous Action Result**: {PAR}
- **Screenshot**: <image>
"""

AITZACTIONPREDICTPROMPTPart2_4_10 = """
### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
- **Current Screen Description**: {SD}
- **Previous Action Result**: {PAR}
- **Screenshot**: <image>
"""


AITZACTIONPREDICTPROMPTPart3_1_2_3_4 = """
## Output Format
You are required to response in the format below, consisting of 1 part with the following keys and corresponding content, and especially avoid using unnecessary quotation marks or other punctuation marks.

Action Decision: <Specify the next single step action that make progress towards the success of the user request.>

**Important**: Do not include any output outside of the specified format.

## Output Example

Action Decision: ...
"""

AITZACTIONPREDICTPROMPTPart3_5 = """
## Output Format
- You are required to response in a JSON format, consisting of 2 distinct parts with the following keys and corresponding content:
{
    "Action Think": <Analyze the logic behind your the next single-step action and your future action plan to fulfill the user request.>, 
    "Action Decision": <Specify the next single step action that make progress towards the success of the user request.>
}
**Important**: Do not include any output outside of the specified JSON format.

## Output Example
{
    "Action Think": "...", 
    "Action Decision": "..."
}
"""

AITZACTIONPREDICTPROMPTPart3_6 = """
## Output Format
- You are required to response in a JSON format, consisting of 2 distinct parts with the following keys and corresponding content:
{
    "Next Action Description": <Detail the list of future actions to complete the user request.>,
    "Action Decision": <Specify the next single step action that make progress towards the success of the user request.>
}
**Important**: Do not include any output outside of the specified JSON format.

## Output Example
{
    "Next Action Description": "...",
    "Action Decision": "..."
}
"""



AITZACTIONPREDICTPROMPTPart3_7_8_9_10 = """
## Output Format
- You are required to response in a JSON format, consisting of 3 distinct parts with the following keys and corresponding content:
{
    "Action Think": <Analyze the logic behind your the next single-step action and your future action plan to fulfill the user request.>, 
    "Next Action Description": <Detail the list of future actions to complete the user request.>,
    "Action Decision": <Specify the next single step action that make progress towards the success of the user request.>
    "Action Result": <Predict the action outcomes after executing the decided action>
}
**Important**: Do not include any output outside of the specified JSON format.

## Output Example
{
    "Action Think": "...", 
    "Next Action Description": "...",
    "Action Decision": "..."
}
"""

AITZACTIONPREDICTOUTPUT_1_2_3_4 = Template("Action Decision: $action")

AITZACTIONPREDICTOUTPUT_5 = Template("""
{
    "Action Think": "$actionThink", 
    "Action Decision": "$action"
}
""")

AITZACTIONPREDICTOUTPUT_6 = Template("""
{
    "Next Action Description": "$actionDesc",
    "Action Decision": "$action"
}
""")

AITZACTIONPREDICTOUTPUT_7_8_9_10 = Template("""
{
    "Action Think": "$actionThink", 
    "Next Action Description": "$actionDesc",
    "Action Decision": "$action"
}
""")

AblationPromptsPart12 = {
    1: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_1_5_6_7,
    2: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_2_8,
    3: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_3_9, 
    4: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_4_10, 
    5: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_1_5_6_7, 
    6: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_1_5_6_7, 
    7: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_1_5_6_7, 
    8: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_2_8, 
    9: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_3_9, 
    10: AITZACTIONPREDICTPROMPTPart1 + AITZACTIONPREDICTPROMPTPart2_4_10, 
}

AblationPromptsPart3 = {
    1: AITZACTIONPREDICTPROMPTPart3_1_2_3_4,
    2: AITZACTIONPREDICTPROMPTPart3_1_2_3_4,
    3: AITZACTIONPREDICTPROMPTPart3_1_2_3_4,
    4: AITZACTIONPREDICTPROMPTPart3_1_2_3_4,
    5: AITZACTIONPREDICTPROMPTPart3_5, 
    6: AITZACTIONPREDICTPROMPTPart3_6, 
    7: AITZACTIONPREDICTPROMPTPart3_7_8_9_10, 
    8: AITZACTIONPREDICTPROMPTPart3_7_8_9_10, 
    9: AITZACTIONPREDICTPROMPTPart3_7_8_9_10, 
    10: AITZACTIONPREDICTPROMPTPart3_7_8_9_10, 
}

AblationOutputs = {
    1: AITZACTIONPREDICTOUTPUT_1_2_3_4, 
    2: AITZACTIONPREDICTOUTPUT_1_2_3_4,
    3: AITZACTIONPREDICTOUTPUT_1_2_3_4,
    4: AITZACTIONPREDICTOUTPUT_1_2_3_4,
    5: AITZACTIONPREDICTOUTPUT_5,
    6: AITZACTIONPREDICTOUTPUT_6,
    7: AITZACTIONPREDICTOUTPUT_7_8_9_10,
    8: AITZACTIONPREDICTOUTPUT_7_8_9_10,
    9: AITZACTIONPREDICTOUTPUT_7_8_9_10,
    10: AITZACTIONPREDICTOUTPUT_7_8_9_10,
}

def convertAction(item:dict, mode="train"):
    action_type = item['result_action_type']
    if action_type == 3:
        action = f"TYPE [{item['result_action_text']}]"
    elif action_type == 4:
        # Extract coordinates
        touch_coords = eval(item['result_touch_yx'])
        lift_coords = eval(item['result_lift_yx'])
        touch_y, touch_x = touch_coords
        lift_y, lift_x = lift_coords
                
        # Check if it's a click (same coordinates)
        if (touch_x == lift_x and touch_y == lift_y) or math.sqrt((touch_x - lift_x)**2 + (touch_y - lift_y)**2) < 0.04:
            imagePath = os.path.join(f"./GUIData/android_in_the_zoo/{mode}/images", os.path.basename(item["image_path"]))
            image = Image.open(imagePath)
            width, height = image.size
            bboxes = eval(item["ui_positions"])
            # convertBboxes = [[x, y , (x + w), (y + h)] for y, x, h, w in bboxes]
            convertBboxes = [[x / width * 1000, y / height * 1000, (x + w) / width * 1000, (y + h) / height * 1000] for y, x, h, w in bboxes]
            for bbox in convertBboxes:
                x1, y1, x2, y2 = bbox
                if x1 <= touch_x * 1000 <= x2 and y1 <= touch_y * 1000 <= y2:
                    touch_x = (x1 + x2) / 2000
                    touch_y = (y1 + y2) / 2000
            
            action = f"CLICK <point>[[{touch_x * 1000:.1f}, {touch_y * 1000:.1f}]]</point>"
        # elif math.sqrt((touch_x - lift_x)**2 + (touch_y - lift_y)**2) < 0.04:
        #     action = f"CLICK <point>[[{touch_x*1000:.1f}, {touch_y*1000:.1f}]]</point>"
        else:
            # Calculate deltas
            delta_x = abs(touch_x - lift_x)
            delta_y = abs(touch_y - lift_y)
            
            if delta_y > delta_x:
                if lift_y < touch_y:
                    action = "SCROLL [UP]"
                else:
                    action = "SCROLL [DOWN]"
            else:
                if lift_x < touch_x:
                    action = "SCROLL [LEFT]"
                else:
                    action = "SCROLL [RIGHT]"
                    
    elif action_type == 5:
        action = "PRESS_BACK"
    elif action_type == 6:
        action = "PRESS_HOME"
    elif action_type == 7:
        action = "ENTER"
    elif action_type == 10:
        action = "COMPLETE"
    elif action_type == 11:
        action = "IMPOSSIBLE"
    else:
        raise NotImplementedError(f"{action_type} cannot be parsed")
    return action

def preprocessEpisode(jsonPath, ablation:int, mode="train"):
    preprocessedData = []
    with open(jsonPath, "r") as f:
        data = json.load(f)
        
    actionStr = [convertAction(item, mode=mode) for item in data]
    actionDescs = [item.get("coat_action_desc","") for item in data]
    actionResults = [item.get("coat_action_result","") for item in data]
    for i, item in enumerate(data):
        imagePath = os.path.join(f"./GUIData/android_in_the_zoo/{mode}/images", os.path.basename(item["image_path"]))
        image = Image.open(imagePath)
        width, height = image.size
        bboxes = eval(item["ui_positions"])
        convertBboxes = [[x / width * 1000, y / height * 1000, (x + w) / width * 1000, (y + h) / height * 1000] for y, x, h, w in bboxes]
        
        screenDesc = item.get("coat_screen_desc","")
        actionThink = item.get("coat_action_think","")
        actionDesc = actionDescs[i]
        action = actionStr[i]
        actionResult = actionResults[i]
        previousActionDescs = actionDescs[:i]
        previousActionResult = actionResults[i - 1] if i > 0 else ""
        
        
        sample = copy.deepcopy(AITZACTIONPREDICTSAMPLE)
        sample["episodeID"] = item["episode_id"]
        sample["stepID"] = item["step_id"]
        sample["images"].append(imagePath)
        sample["width"], sample["height"] = width, height
        sample["bboxes"] = convertBboxes
        sample["screenDesc"] = screenDesc
        sample["previousActionDescs"] = previousActionDescs
        sample["previousActionResult"] = previousActionResult
        sample["actionThink"] = actionThink
        sample["actionThink"] = actionThink
        sample["actionDesc"] = actionDesc
        sample["action"] = action
        sample["actionResult"] = actionResult
        
        prompt12 = AblationPromptsPart12[ablation]
        prompt3 = AblationPromptsPart3[ablation]
        output = AblationOutputs[ablation]
        
        sample["messages"][0]["content"] = prompt12.format(
            finalGoal=item["instruction"],
            PAD=previousActionDescs,
            PAR = previousActionResult,
            SD = screenDesc
        ) + prompt3
        
        sample["messages"][1]["content"] = output.substitute(
            actionThink=actionThink,
            actionDesc=actionDesc,
            action=action,
            actionResult=actionResult
        )
        
        
        preprocessedData.append(sample)
        
    return preprocessedData

if __name__ == "__main__":
    for ablation in tqdm(range(1, 11), total=10):
        trainData, testData = [], []
        for jsonFile in os.listdir("./GUIData/android_in_the_zoo/train/jsons"):
            jsonPath = os.path.join("./GUIData/android_in_the_zoo/train/jsons", jsonFile)
            preprocessData = preprocessEpisode(jsonPath, ablation=ablation,  mode="train")
            trainData.extend(preprocessData)
        
        
        for jsonFile in os.listdir("./GUIData/android_in_the_zoo/test/jsons"):
            jsonPath = os.path.join("./GUIData/android_in_the_zoo/test/jsons", jsonFile)
            preprocessData = preprocessEpisode(jsonPath, ablation=ablation, mode="test")
            testData.extend(preprocessData)
            
            
        with open(f'./data/GUI/aitz_ablation_{ablation}_action_predict_train.json', "w", encoding="utf-8") as f:
            print(f'dump {len(trainData)} train samples')
            json.dump(trainData, f, ensure_ascii=False, indent=4)
            
        with open(f'./data/GUI/aitz_ablation_{ablation}_action_predict_test.json', "w", encoding="utf-8") as f:
            print(f'dump {len(testData)} test samples')
            json.dump(testData, f, ensure_ascii=False, indent=4)
        