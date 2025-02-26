import os, json, tqdm, re, copy
from templates import ACTIONPREDICTSAMPLE, ANDORIDCONTROLHIGHACTIONPREDICTPROMPT
import random
from tqdm import tqdm

jsonPath = "GUIData/android_control/jsons"
imagePath = "GUIData/android_control/images"
layoutPath = "GUIData/android_control/layouts"

with open('android_control_splits.json', 'r') as f:
    splits = json.load(f)
trainID = splits["train"] + splits["validation"]
testID = splits["test"]
allID = trainID + testID


def __in_bbox(x, y, bbox):
    return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]

def __get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def action2str(action:dict, layoutPath:str, width:int, height:int) -> str:
    actionType = action["action_type"]
    with open(layoutPath, "r", encoding="utf-8") as f:
        layouts = [json.loads(line.strip()) for line in f]

    
    if actionType == "click":
        x, y = action["x"], action["y"]
        for layout in layouts:
            if layout["is_clickable"] and "bbox_pixels" in layout and __in_bbox(x, y, layout["bbox_pixels"]):
                center_x, center_y = __get_bbox_center(layout["bbox_pixels"])
                x, y = center_x, center_y
                break
        ux, uy = (x / width) * 1000, (y / height) * 1000 
        result = f"CLICK <point>[[{ux:.1f}, {uy:.1f}]]</point>"
    elif actionType == "open_app":
        result = f"OPENAPP {action['app_name']}"
    elif actionType == "input_text":
        result = f"TYPE [{action['text']}]"
    elif actionType == "scroll":
        result = f"SCROLL [{action['direction'].upper()}]"
    elif actionType == "wait":
        result = "WAIT"
    elif actionType == "navigate_home":
        result = "PRESS_HOME"
    elif actionType == "navigate_back":
        result = "PRESS_BACK"   
    elif actionType == "long_press":
        x, y = action["x"], action["y"]
        for layout in layouts:
            if layout["is_clickable"] and "bbox_pixels" in layout and __in_bbox(x, y, layout["bbox_pixels"]):
                center_x, center_y = __get_bbox_center(layout["bbox_pixels"])
                x, y = center_x, center_y
                break
        ux, uy = (x / width) * 1000, (y / height) * 1000 
        result = f"LONG_CLICK <point>[[{ux:.1f}, {uy:.1f}]]</point>"
    return result


        
def preprocessEpisode(episodeID:int):
    preprocessData = []
    metaDataPath = os.path.join(jsonPath, f"metadata_episode_{episodeID}.json")
    with open(metaDataPath, "r") as f:
        metaData = json.load(f)
    heights, widths = metaData["screenshot_heights"], metaData["screenshot_widths"]
    actions, screenshots, layouts = metaData["actions"], metaData["screenshots"], metaData["accessibility_trees"]
    step_instructions = metaData["step_instructions"]
    screenshots = [os.path.join(imagePath, f"episode_{episodeID}_{os.path.basename(screenshot)}") for screenshot in screenshots]
    layouts = [os.path.join(layoutPath, f"episode_{episodeID}_{os.path.basename(layout)}") for layout in layouts]
    assert (len(actions) == len(screenshots) - 1) and (len(screenshots) == len(layouts))
    
    actionStrs = [action2str(actions[i], layouts[i], widths[i], heights[i]) for i in range(len(actions))]
    
    for i in range(len(actions) + 1):
        sample = copy.deepcopy(ACTIONPREDICTSAMPLE)
        prompt = copy.deepcopy(ANDORIDCONTROLHIGHACTIONPREDICTPROMPT)
        sample["images"].append(screenshots[i])
        sample["layouts"].append(layouts[i])
        
        previousActions = step_instructions[:i]
        sample["messages"][0]["content"] = prompt.format_map({"finalGoal": metaData["goal"], "previousActions":previousActions})
        sample["episodeID"] = metaData["episode_id"]
        sample["stepID"] = i + 1
        sample["width"] = widths[i]
        sample["height"] = heights[i]
        
        labelAction = actionStrs[i] if i != len(actions) else "COMPLETE"
        
        sample["messages"][1]["content"] = f"action: {labelAction}"
        
        preprocessData.append(sample)
    return preprocessData


if __name__ == "__main__":
    random.seed(3407)
    trainData, testData = [], []
    for episodeId in tqdm(trainID, total=len(trainID), desc="train"):
        trainData.extend(preprocessEpisode(episodeId))
    for episodeId in tqdm(testID, total=len(testID), desc="test"):
        testData.extend(preprocessEpisode(episodeId))
        
    with open('./data/GUI/android_control_high_action_predict_train.json', "w", encoding="utf-8") as f:
        print(f'dump {len(trainData)} train samples')
        json.dump(trainData, f, ensure_ascii=False, indent=4)
        
    with open('./data/GUI/android_control_high_action_predict_test.json', "w", encoding="utf-8") as f:
        print(f'dump {len(testData)} test samples')
        json.dump(testData, f, ensure_ascii=False, indent=4)
        
        