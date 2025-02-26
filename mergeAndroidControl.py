import json, os, shutil
from tqdm import tqdm

dataRootPath = "./GUIdata/android_control_parsed/"
jsonPath = "./GUIdata/android_control/jsons"
imagePath = "./GUIdata/android_control/images"
layoutPath = "./GUIdata/android_control/layouts"

for path in [jsonPath, imagePath, layoutPath]:
    os.makedirs(path, exist_ok=True)

def preprocessEpisode(episodeID:int):
    episodePath = os.path.join(dataRootPath, f"episode_{episodeID}")
    metaDataPath = os.path.join(episodePath, f"metadata_episode_{episodeID}.json")
    
    shutil.copy(metaDataPath, os.path.join(jsonPath, f"metadata_episode_{episodeID}.json"))
    
    for imageFile in os.listdir(os.path.join(episodePath, "screenshots")):
        src = os.path.join(episodePath, "screenshots", imageFile)
        dst = os.path.join(imagePath, f"episode_{episodeID}_{imageFile}")
        shutil.copy(src, dst)
    
    for layoutFile in os.listdir(os.path.join(episodePath, "accessibility_trees")):
        src = os.path.join(episodePath, "accessibility_trees", layoutFile)
        dst = os.path.join(layoutPath, f"episode_{episodeID}_{layoutFile}")
        shutil.copy(src, dst)


with open('./GUIdata/android_control_splits.json', 'r') as f:
    splits = json.load(f)
trainID = splits["train"] + splits["validation"]
testID = splits["test"]
allID = trainID + testID

for episodeID in tqdm(allID, total=len(allID)):
    preprocessEpisode(episodeID)