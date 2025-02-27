from openai import OpenAI
import os, json, base64, re, copy, openai
from tqdm import tqdm
from templates import QUERYPOLISHPROMPT, QUERYFVALIDATEPROMPT, QUERYINFERENCESAMPLE

jsonPath = "./GUIData/UIBert/uibert_raw.json"
imageRootPath = "./GUIData/UIBert/"

client = OpenAI(
    api_key="your key", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



X, Y = 1080, 1920

def polishQuery(gtBbox, rawQuery, imagePath:str):
    with open(imagePath, "rb") as f:
        imageBase64 = base64.b64encode(f.read()).decode('utf-8')
    message = [
        {
            "role": "user",
            "content":[
                {
                    "type":"text",
                    "text":QUERYPOLISHPROMPT.format_map({"query":rawQuery, "bbox":gtBbox})
                }, 
                {
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/png;base64,{imageBase64}"
                    }
                }
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=message,
        temperature=0, 
        top_p=1,
        n=1
    )
    
    polishedQuery = completion.choices[0].message.content
    return polishedQuery

def reGrounding(imagePath:str, polishedQuery:str):
    with open(imagePath, "rb") as f:
        imageBase64 = base64.b64encode(f.read()).decode('utf-8')
    message = [
        {
            "role": "user",
            "content":[
                {
                    "type":"text",
                    "text":QUERYFVALIDATEPROMPT.format_map({"query":polishedQuery})
                }, 
                {
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/png;base64,{imageBase64}"
                    }
                }
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=message,
        temperature=0, 
        top_p=1,
        n=1
    )
    
    pred = completion.choices[0].message.content
    
    match = re.search(r'bbox: \[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]', pred)

    predBbox = [float(match.group(i)) for i in range(1, 5)] if match is not None else [0, 0, 0, 0]
    
    return predBbox

def validateBbox(gtBbox, predBbox):
    centerx, centery = (predBbox[0] + predBbox[2]) / 2, (predBbox[1] + predBbox[3]) / 2
    
    return gtBbox[0] <= centerx <= gtBbox[2] and gtBbox[1] <= centery <= gtBbox[3]
    

with open(jsonPath, "r") as f:
    jsonData = json.load(f)
    
processedData = []
stopImage = None
processed = 0
savePath = './data/GUI/uibert_query_summary_polished_preprocessed.json'
if os.path.exists(savePath):
    with open(savePath, "r") as f:
        processedData = json.load(f)
    processedData[-1]["images"][0]    


    for stopPoint, imageMeta in enumerate(jsonData):
        if os.path.abspath(os.path.normpath(os.path.join(imageRootPath, imageMeta["img_filename"]))) == os.path.abspath(os.path.normpath(stopImage)):
            processed = stopPoint
            break

    
for i, imageMeta in tqdm(enumerate(jsonData[processed:]), total=len(jsonData) - processed, desc="polishing UIBERT"):
    for elements in imageMeta["elements"]:
        imagePath = os.path.join(imageRootPath, imageMeta["img_filename"])
        rawQuery = elements["instruction"]
        try:
            polishedQuery = polishQuery(elements["bbox"], rawQuery, imagePath)
            predBbox = reGrounding(imagePath, polishedQuery)
        except openai.BadRequestError:
            continue
        if validateBbox(gtBbox=elements["bbox"], predBbox=predBbox):
            sample = copy.deepcopy(QUERYINFERENCESAMPLE)
            sample["images"].append(imagePath)
            x1, y1, x2, y2 = elements["bbox"]
            x1, y1, x2, y2 = x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000
            bbox = f"<|box start|>({x1: .1f},{y1: .1f}),({x2 : .1f},{y2 : .1f})<|box end|>".strip().replace(" ", "")
            sample["messages"][0]["content"] = sample["messages"][0]["content"].format_map({"bbox":bbox})
            sample["messages"][1]["content"] = sample["messages"][1]["content"].format_map({"query":polishedQuery})
            sample["normBbox"] = elements["bbox"]
            sample["normPredBbox"] = predBbox
            sample["bbox"] = bbox
            sample["polishedQuery"] = polishedQuery
            sample["rawQuery"] = rawQuery
            processedData.append(sample)
        if i % 50 == 0:
            with open('./data/GUI/uibert_query_summary_polished_preprocessed.json', "w", encoding="utf-8") as f:
                print(f'dump {len(processedData)} samples')
                json.dump(processedData, f, ensure_ascii=False, indent=4)


with open('./data/GUI/uibert_query_summary_polished_preprocessed.json', "w", encoding="utf-8") as f:
    print(f'dump {len(processedData)} samples')
    json.dump(processedData, f, ensure_ascii=False, indent=4)