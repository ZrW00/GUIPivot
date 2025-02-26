from string import Template

SAMPLE = {
    "episodeID":None,
    "messages": [
        {
            "role": "user",
            "content": ""
        },
        {
            "role": "assistant",
            "content": "action: {action}"
        }
    ],
    "images": [],
    "xmls": []
}

SYSPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 4: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. 

And your previous actions, current task instruction, previous actions and associated screenshot are as follows:

Final goal: {finalGoal}
Previous actions: {previousActions}
Screenshot: <image>

Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks. (where osatlas action must be one of the action formats I provided):

action: 
"""


QUERYPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish the following task.

Given the following UI screenshot:
<image>.

And the action:
"CLICK on the item within the bounding box {bbox}."

Your task is to generate the most likely query that corresponds to the provided action in the context of the UI screenshot. 

# Instructions
Before generating the query, carefully analyze the bounding box {bbox} and its relationship to the surrounding content by considering the following:

1. Bounding box location: Precisely identify the region highlighted by {bbox} in the screenshot. Pay special attention to its position relative to the entire UI layout.
2. Bounding box content: Understand the content (e.g., text, icons, or images) located within {bbox}, ensuring it matches the intended clickable element.
3. Contextual relevance: Analyze the relationship between {bbox} and nearby UI elements to infer its purpose.
4. Task intent: Align the query with the implied goal or action associated with the specified bounding box.

# Output Format
Your output must be a clear, concise query that reflects the intent to interact with the UI element inside the bounding box. Follow the strict format below without unnecessary punctuation or quotation marks:

query: 
"""

QUERYINFERENCEPOINTPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish the following task.

Given the following inputs:
1. UI screenshot:
<image>.

2. Action:
"CLICK on the coordinate {point}."

Your task is to generate the most likely query that corresponds to the provided action in the context of the UI screenshot.

# Instructions
1. Analyze the Coordinates
- Carefully consider the coordinates {point} provided in the action.
- Analyze its relationship with the surrounding UI elements and how the coordinate corresponds to specific interactive elements like buttons, text fields, icons, etc.

2. Understand the Intent
- The query should reflect the purpose of the click, such as selecting an option, clicking a button, or interacting with a visual element in the UI.

3. Generate the Query
- Ensure the query is clear, concise, and directly related to the UI element located at the specified coordinate {point}.

# Output Format
Your output must be a clear, concise query that reflects the intent to interact with the UI element on the coordinate {point}. Follow the strict format below without unnecessary punctuation or quotation marks:

query: 
"""

QUERYINFERENCESAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": QUERYPROMPT
        },
        {
            "role": "assistant",
            "content": "query: {query}"
        }
    ],
    "images": [],
    "layouts":[]
}


GROUNDINGPROMPT = """
You are now operating in Grounding Mode. Your primary goal is to help users accurately map commands to UI elements.

# Task Overview
Given the following inputs:
1. UI screenshot:
<image>.
2. Query command:
"{query}"

Your goal is to generate the bounding box position of the UI element that corresponds most accurately to the action described in the query command. Output the bounding box coordinates in the following format: 

bbox: <|box_start|>(x1,y1),(x2,y2)<|box_end|>

# Instructions for Generating the Bounding Box
To improve grounding accuracy, carefully follow these steps:

1. Understand the Task
- Analyze the query command to identify the described action (e.g., "open daily recommendations," "enter the search interface").
- Interpret both explicit details and implicit aspects of the command to infer the user's intent.

2. Locate Relevant UI Elements
- Examine the UI screenshot to identify the element(s) matching the description in the query.
- Leverage visual context clues such as labels, icons, colors, and layout to pinpoint the most relevant target.

3. Ensure Bounding Box Precision
- Ensure the bounding box tightly encompasses the identified UI element.
- Verify that the coordinates align precisely with the element's edges and exclude any unnecessary padding or unrelated elements.

4. Maintain Contextual Consistency
- Consider the overall UI layout to ensure the bounding box aligns with the user's intent and the query's context.
- Resolve ambiguities by inferring the user's intent based on both the UI structure and the action described in the query.

# Output Guidelines
Ensure the coordinates are as precise as possible to match the area defined by the query command. Your output must follow this exact format for the bounding box (e.g. bbox: <|box_start|>(x1,y1),(x2,y2)<|box_end|>) without unnecessary punctuation or quotation marks:

bbox: 
"""


GROUNDINGSAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": GROUNDINGPROMPT
        },
        {
            "role": "assistant",
            "content": "bbox: {bbox}"
        }
    ],
    "images": [],
}



ACTIONPREDICTPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 4: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

Custom Action 5: OPENAPP
    - purpose: Open an app.
    - format: OPENAPP <APP_NAME>
    - example usage: OPENAPP Zoho Meeting

Custom Action 6: WAIT
    - purpose: Wait a set number of seconds for something on screen (e.g., a loading bar).
    - format: WAIT
    - example usage: WAIT

Custom Action 7: LONG_CLICK
    - purpose: Long click at the specified position.
    - format: LONG_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_CLICK <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. 

And your final goal, previous actions and associated screenshot are as follows:

Final goal: {finalGoal}
Previous actions: {previousActions}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal, previous actions, and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.

# Output Format
Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks. (where osatlas action must be one of the action formats I provided):

action: 
"""

ANDORIDCONTROLLOWACTIONPREDICTPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 4: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

Custom Action 5: OPENAPP
    - purpose: Open an app.
    - format: OPENAPP <APP_NAME>
    - example usage: OPENAPP Zoho Meeting

Custom Action 6: WAIT
    - purpose: Wait a set number of seconds for something on screen (e.g., a loading bar).
    - format: WAIT
    - example usage: WAIT

Custom Action 7: LONG_CLICK
    - purpose: Long click at the specified position.
    - format: LONG_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_CLICK <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning, follow current step instruction to determine the most appropriate next action. 

And your final goal, previous actions, current step instruction, and associated screenshot are as follows:

Final goal: {finalGoal}
Previous actions: {previousActions}
Current step instruction: {actionDesc}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal, previous actions, and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.
- Ensure that your action suggestion is consistent with the desired outcome based on previous steps.

# Output Format
Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks. (where osatlas action must be one of the action formats I provided):

action: 
"""


ANDORIDCONTROLHIGHACTIONPREDICTPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 4: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

Custom Action 5: OPENAPP
    - purpose: Open an app.
    - format: OPENAPP <APP_NAME>
    - example usage: OPENAPP Zoho Meeting

Custom Action 6: WAIT
    - purpose: Wait a set number of seconds for something on screen (e.g., a loading bar).
    - format: WAIT
    - example usage: WAIT

Custom Action 7: LONG_CLICK
    - purpose: Long click at the specified position.
    - format: LONG_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_CLICK <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. 

And your final goal, previous actions and associated screenshot are as follows:

Final goal: {finalGoal}
Previous actions: {previousActions}
Screenshot: <image>

# Instructions for Determining the Next Action
- Carefully analyze the final goal, previous actions, and the current screenshot.
- Identify the most suitable action based on the context and the goal.
- Make sure the action you suggest aligns with the desired outcome, considering the previous steps.

# Output Format
Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks. (where osatlas action must be one of the action formats I provided):

action: 
"""


ACTIONPREDICTSAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": ACTIONPREDICTPROMPT
        },
        {
            "role": "assistant",
            "content": None
        }
    ],
    "images": [],
    "layouts":[],
    "episodeID":None,
    "stepID":None,
    "width":None,
    "height":None
}



AITZACTIONPREDICTPROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>

Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - Purpose: SCROLL in the specified direction.
    - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - Example Usage: SCROLL [UP]

2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 4: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 5: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the goal instruction and action history, then perform reasoning to determine the most appropriate next action. 

### Task Execution Guidelines
- Carefully evaluate the task goal, previous actions, and the current screen description.
- Check whether previous actions have fulfilled the user request.
- Analyze visible apps, icons, and buttons on the current screen that are relevant to the user request.
- Formulate a logical plan for the next action, avoiding unnecessary or redundant steps.



### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
- **Current Screen Description**: {SD}
- **Previous Action Result**: {PAR}
- **Screenshot**: <image>

## Output Format
- You are required to response in a JSON format, consisting of 4 distinct parts with the following keys and corresponding content:
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
    "Action Decision": "...",
    "Action Result": "..."
}
"""

AITZACTIONPREDICTPROMPTPart1 = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>

Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - Purpose: SCROLL in the specified direction.
    - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - Example Usage: SCROLL [UP]

2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 4: IMPOSSIBLE
    - purpose: Indicate the task is impossible.
    - format: IMPOSSIBLE
    - example usage: IMPOSSIBLE

Custom Action 5: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the goal instruction and action history, then perform reasoning to determine the most appropriate next action. 
"""

AITZACTIONPREDICTPROMPTPart2 = """

### Final Input Details
Your final goal, previous actions, current screen description, and any additional context are provided as follows:
- **Final Goal**: {finalGoal}
- **Previous Action Descriptions**: {PAD}
- **Current Screen Description**: {SD}
- **Previous Action Result**: {PAR}
- **Screenshot**: <image>
"""

AITZACTIONPREDICTPROMPTPart3 = """

## Output Format
- You are required to response in a JSON format, consisting of 4 distinct parts with the following keys and corresponding content:
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
    "Action Decision": "...",
    "Action Result": "..."
}
"""


AITZACTIONPREDICTOUTPUT = Template("""
{
    "Action Think": "$actionThink", 
    "Next Action Description": "$actionDesc",
    "Action Decision": "$action",
    "Action Result": "$actionResult"
}
""")


AITZACTIONPREDICTSAMPLE = {
    "messages": [
        {
            "role": "user",
            "content": AITZACTIONPREDICTPROMPT
        },
        {
            "role": "assistant",
            "content": None
        }
    ],
    "images": [],
    "episodeID":None,
    "stepID":None,
    "width":None,
    "height":None,
    "bboxes":None,
    "screenDesc":None,
    "previousActionDescs":None,
    "previousActionResult":None,  
    "actionThink":None,
    "actionDesc":None,
    "action":None,
    "actionResult":None
}

QUERYPOLISHPROMPT = """
# Task Overview
I will provide you with a mobile screenshot, the user query, and the bounding box (bbox). The bbox is given in the format [x1, y1, x2, y2], where (x1, y1) represents the top-left corner and (x2, y2) represents the bottom-right corner of the bounding box. All coordinates are normalized to the range [0, 1].

- Mobile Screenshot: [screenshot]
- User Query: {query}
- Bounding Box (bbox): {bbox}

Please polish the user query into the following format:
"click on the [element_name] for [purpose]"

# Instructions

1. **Identify the Element:**
   - Identify the element in the screenshot that corresponds to the given bbox.
   - Determine the name of the element based on its appearance and context.

2. **Determine the Purpose:**
   - Understand the purpose or action associated with the element from the user query.

3. **Rewrite the Query:**
   - Rewrite the user query in the specified format: "click on the [element_name] for [purpose]".


# Example
Input:
- Screenshot: [screenshot]
- Query: "Press the submit button to send the form"
- Bounding Box: [0.2, 0.3, 0.4, 0.5]
Output:
- "click on the submit button for sending the form"

# Output Guidelines
- Your output should not contain unnecessary punctuation or quotation marks.
"""

QUERYFVALIDATEPROMPT = """
You are now operating in Grounding Mode. I will provide you with a mobile screenshot and the user query. Your primary goal is to help users accurately map user query commands to UI elements.

# Task Overview
Given the following inputs:
1. UI screenshot:
[screenshot]
2. Query command:
"{query}"

Your goal is to generate the normalized bounding box position of the UI element that corresponds most accurately to the action described in the query command. Output the bounding box coordinates within [0, 1]. 

# Bounding Box Normalization
The bounding box coordinates should be normalized to the range [0,1]. To achieve this, divide the pixel values of each coordinate by the width and height of the screenshot, respectively.

# Output Format
bbox: [x1,y1,x2,y2]


# Instructions for Generating the Bounding Box
To improve grounding accuracy, carefully follow these steps:

1. Understand the Task
- Analyze the query command to identify the described action (e.g., "open daily recommendations," "enter the search interface").
- Interpret both explicit details and implicit aspects of the command to infer the user's intent.

2. Locate Relevant UI Elements
- Examine the UI screenshot to identify the element(s) matching the description in the query.
- Leverage visual context clues such as labels, icons, colors, and layout to pinpoint the most relevant target.

3. Ensure Bounding Box Precision
- Ensure the bounding box tightly encompasses the identified UI element.
- Verify that the coordinates align precisely with the element's edges and exclude any unnecessary padding or unrelated elements.

4. Maintain Contextual Consistency
- Consider the overall UI layout to ensure the bounding box aligns with the user's intent and the query's context.
- Resolve ambiguities by inferring the user's intent based on both the UI structure and the action described in the query.

# Output Guidelines
Ensure the coordinates are as precise as possible to match the area defined by the query command. Your output must follow this exact format for the bounding box (e.g. bbox: [x1,y1,x2,y2]) without unnecessary punctuation or quotation marks:

bbox: 
"""



QUERYPOLISHPOINTPROMPT = """
# Task Overview
I will provide you with a mobile screenshot, the user query, and the click coordinate. The coordinate is given in the format (x,y) and is normalized to the range [0, 1].


- Mobile Screenshot: [screenshot]
- User Query: {query}
- Click Coordinate: {coordinate}

Please polish the user query into the following format:
"click on the [element_name] for [purpose]"

# Instructions

1. **Identify the Element:**
   - Examine the screenshot to identify the element that corresponds to the provided click coordinate.
   - Determine the name of the element based on its visual appearance and context within the UI.

2. **Determine the Purpose:**
   - Analyze the user query to understand the action or purpose associated with the element (e.g., submitting a form, opening a menu, etc.).

3. **Rewrite the Query:**
   - Rewrite the user query to follow this specific format: "click on the [element_name] for [purpose]".
   - Ensure that the rewritten query accurately reflects the element's name and its purpose.

# Example
Input:
- Screenshot: [screenshot]
- Query: "Press the submit button to send the form"
- Click Coordinate: (0.2, 0.3)

Output:
- "click on the submit button for sending the form"

# Output Guidelines
- Your output should not contain unnecessary punctuation or quotation marks.
"""


QUERYFVALIDATEPOINTPROMPT = """
You are now operating in Grounding Mode. I will provide you with a mobile screenshot and the user query. Your primary goal is to help users accurately map user query commands to UI elements.

# Task Overview
Given the following inputs:
1. UI screenshot:
   [screenshot]
2. Query command:
   "{query}"

Your goal is to generate the click coordinate of the UI element that corresponds most accurately to the action described in the query command. Output the click coordinate within [0, 1].

# Output Format
(x,y)

# Instructions for Generating the Click Coordinate
To improve grounding accuracy, carefully follow these steps:

1. **Understand the Task**
   - Analyze the query command to identify the described action (e.g., "open daily recommendations," "enter the search interface").
   - Interpret both explicit details (keywords such as "open," "click," "select") and implicit aspects (user intentions, like clicking a button or accessing a menu) to infer the user's true intent.

2. **Locate Relevant UI Elements**
   - Examine the UI screenshot to identify the element(s) that match the description in the query.
   - Look for clues such as icons, buttons, labels, text fields, or images that align with the action described.
   - Consider UI layout and visual design to locate the most probable clickable element.

3. **Ensure Click Coordinate Precision**
   - Once the relevant element is identified, determine its exact location within the screenshot.
   - Map the position of the element into normalized coordinates within the [0, 1] scale (where (0, 0) is the top-left corner and (1, 1) is the bottom-right corner).
   - Ensure that the output coordinate reflects the center or appropriate interaction point within the UI element.

4. **Maintain Contextual Consistency**
   - Make sure the selected coordinate aligns with the broader context of the UI. For instance, if the query asks for a button click, ensure the button is in an actionable state and is appropriately positioned within the screenshot's layout.

# Output Guidelines
- Ensure the coordinates are as precise as possible to match the area defined by the query command.
- Your output must follow this exact format for the click coordinate (e.g., (x,y)) without unnecessary punctuation or quotation marks.
"""

QUERYINFERENCEPOLISHPROMPT = """
You are now operating in Executable Language Grounding mode. Your task is to help users generate a query based on the provided UI screenshot and action.

Given the following UI screenshot:
<image>.

And the action:
"CLICK on the item within the bounding box {bbox}."

# Instructions
Follow these steps to generate the appropriate query:

1. **Bounding box location**: Precisely identify the region highlighted by {bbox} in the screenshot. Focus on its position relative to other UI elements.
2. **Bounding box content**: Understand what is located within {bbox}, such as text, icons, or buttons, and confirm it corresponds to the intended clickable element.
3. **Contextual relevance**: Consider how the bounding box relates to surrounding elements to infer its function or role in the UI.
4. **Task intent**: Align the generated query with the implied action associated with the bounding box.

# Output Format:
The query must follow this format:
click on the [element_name] for [purpose]

Ensure the query is concise, clear, and reflects the correct interaction with the UI element inside the bounding box.

Output:
query:
"""





