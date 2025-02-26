from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import re
from .models import Qwen2VLAgent

class ActionType(Enum):
    CLICK=1
    TYPE=2
    SCROLLUP=3
    SCROLLDOWN=4
    SCROLLLEFT=5
    SCROLLRIGHT=6
    PRESSBACK=7
    PRESSHOME=8
    TASKIMPOSSIBLE=9
    TASKCOMPLETE=10
    NONEACTION=11
    WAIT=12
    LONGCLICK=13
    OPENAPP=14
    
    

@dataclass
class AndroidAction():
    action_type: ActionType
    point: Tuple[float, float] = None
    typed_text: str = None
    app_name:str = None

    def __str__(self):
        components = [f"Action Type: {self.action_type.name}"]

        if self.point:
            touch_point_str = f"({self.point[0]:.1f}, {self.self.point[1]:.1f})"
            components.append(f"Click Point: {touch_point_str}")
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")
        return ", ".join(components)

    
def qwen_translate_action(action:str):
    if action == "PRESS_BACK":
        return AndroidAction(action_type=ActionType.PRESSBACK)
    elif action == "PRESS_HOME":
        return AndroidAction(action_type=ActionType.PRESSHOME)
    elif action == "COMPLETE":
        return AndroidAction(action_type=ActionType.TASKCOMPLETE)
    elif action == "IMPOSSIBLE":
        return AndroidAction(action_type=ActionType.TASKIMPOSSIBLE)
    elif action == "SCROLL [RIGHT]":
        return AndroidAction(action_type=ActionType.SCROLLRIGHT)
    elif action == "SCROLL [LEFT]":
        return AndroidAction(action_type=ActionType.SCROLLLEFT)
    elif action == "SCROLL [UP]":
        return AndroidAction(action_type=ActionType.SCROLLUP)
    elif action == "SCROLL [DOWN]":
        return AndroidAction(action_type=ActionType.SCROLLDOWN)
    elif action == "WAIT":
        return AndroidAction(action_type=ActionType.WAIT)
    elif action.startswith("TYPE [") and action.endswith("]"):
        pattern = r"TYPE \[(.*?)\]"
        match = re.search(pattern, action)
        text = match.group(1) if match else ""
        return AndroidAction(action_type=ActionType.TYPE, typed_text=text)
    elif action.startswith("OPENAPP"):
        appName = action.replace("OPENAPP", "").strip().lower()
        return AndroidAction(action_type=ActionType.OPENAPP, app_name=appName)
    elif action.startswith("CLICK <point>[[") and action.endswith("]]</point>"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        return AndroidAction(action_type=ActionType.CLICK, point=(x, y))  
    elif action.startswith("LONG_CLICK <point>[[") and action.endswith("]]</point>"):
        pattern = r"\[\[(-?[0-9.]+),\s*(-?[0-9.]+)\]\]"
        match = re.search(pattern, action)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
        else:
            x, y = None, None
        return AndroidAction(action_type=ActionType.LONGCLICK, point=(x, y))  
    else:
        return AndroidAction(action_type=ActionType.NONEACTION)
        





