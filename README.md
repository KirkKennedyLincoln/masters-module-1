# Module 1: Exploratory Data Analysis on NVIDIA Nemotron-Agentic-Tool-Use-v1

## Project Description
Nemotron-Agent-Tool-Use-v1 dataset from NVIDIA analyzes policy executions
from agentic AI invocations.  It provides every step within the process
including the system prompt, user messages, agent responses, and their
associated tool calls.  

These data points are protected under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Claude AI was used to assist with syntax, proof-reading, and code review during development.

## Dataset Reference
NVIDIA Corporation. (2025). Nemotron-Agentic-Tool-Use-v1 [Data set](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1).
    HuggingFace. https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1

## How to Reproduce
```bash
git clone https://github.com/KirkKennedyLincoln/masters-module-1.git
cd masters-module-1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=agent-eda
jupyter notebook
```
Select the **agent-eda** kernel, then run all cells. The dataset 
downloads automatically if not present.

## Reflection Questions

### Where could poor data cleaning introduce bias?
Bias can enter when parsing the agent's policy since some domains contain substeps that don't conform.

i.e. Healthcare
*Appointment Types*:  
1. **Emergency**: Requires immediate care (tier 1-3)  
2. **Urgent**: Within 24 hours  
3. **Routine**: 7+ day window  
4. **Follow-up**: Linked to previous visit  
i.e. Healthcare
**Cancellation Policy**  
| Time Before Appointment | Fee |  
|-------------------------|-----|  
| <24 hours | $50 |  
| <1 hour | $100 + copay |  
*Exceptions: Medical emergencies (requires MD note)*

We can see that Healthcare tends to include a Cancellation Policy, which is technically counted correctly right now; however, this data could be counted in different ways, which could explain certain things as: higher tool calling, observed inefficiencies, and healthcare having a more difficult escalation path. If we trained agents on some of the statistics from this dataset, then healthcare would be lacking the sufficient parsing to provide sufficient datapoints. Looking back to our "Policy Frequency of Each Domain" healthcare is highly underrepresented and would need some sort of sampling and balancing if used for formal training.

### How would this workflow change for an ML pipeline?
If this data were used for training then we would involve train/test splits so we can separate the data for training/validation.  Prior to that we would ingest and clean the data then hand it off to the training functions.  One hot encoding can be introduced since we have repetitive tool calling within policies if we wanted to train for predictions with principle component analysis using k-means clustering.

When it comes to scaling we would introduce the sister-file that was found named *tool_calling.jsonl*; however, it would require a much larger machine.  There would need to be more analysis on the tool calling dataset in order to verify what features would be best to engineer.

**tool_calling.jsonl example**
```json
{"uuid": "6798bc3f-cec8-41d2-8bc5-47d73b91fe32", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "Predict the change in forest area over the next 5 years without considering human impact."}, {"role": "assistant", "content": "The prediction of forest area change requires a specific geographic location. Please provide the city or region where you'd like to make this prediction.", "tool_calls": null, "reasoning_content": "Okay, let's see. The user wants to predict the change in forest area over the next 5 years without considering human impact. So, first, I need to check which tools are available. There's the predict_forest_area function. Let me look at its parameters. It requires location, num_years, and include_humans. The user didn't specify a location, so I need to figure that out. Wait, maybe they mentioned a city earlier? The current query doesn't include a location. Hmm, but the user's question is general. Wait, the tools also have geocode_city which gets coordinates for a city. But the user hasn't provided a city name. Oh, and the get_ip_zipcode might get a ZIP code from an IP, but that's not directly related to location for the forest prediction. Wait, maybe the user's IP can be used to find their location? But that's a stretch. The problem is the predict_forest_area needs a location. Since the user didn't specify, maybe I should ask for the location. Alternatively, check if there's a way to get the location through other means. But the available tools don't have a function to get the user's current location automatically. The geocode_city requires a city name, which the user hasn't provided. So, in this case, I need to prompt the user to provide the specific location (city) where the prediction should be made. Because without that, the function can't be called. The include_humans is set to False as per the user's request (\"without considering human impact\"), and num_years is 5. So the missing parameter is location. Therefore, the correct step is to ask the user to provide the city or location for the prediction."}], "license": "cc-by-4.0", "used_in": ["nano_v3"], "tools": [{"type": "function", "function": {"name": "get_ip_zipcode", "description": "Retrieves the ZIP code of a given IP address using the ip-api.com API.", "parameters": {"type": "object", "properties": {"ip": {"type": "string", "description": "The IP address to locate."}}, "required": ["ip"]}}}, {"type": "function", "function": {"name": "is_leap_year", "description": "Checks if a year is a leap year.", "parameters": {"type": "object", "properties": {"year": {"type": "integer", "description": "The year to check."}}, "required": ["year"]}}}, {"type": "function", "function": {"name": "predict_forest_area", "description": "Predicts the change in forest area over a specified number of years.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The geographic location for the prediction."}, "num_years": {"type": "integer", "description": "The number of years into the future to make the prediction."}, "include_humans": {"type": "boolean", "description": "Whether to include the impact of human activities in the prediction. Defaults to False."}}, "required": ["num_years", "include_humans", "location"]}}}, {"type": "function", "function": {"name": "geocode_city", "description": "Retrieves the latitude and longitude coordinates of a given city using the Maps.co Geocoding API.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The name of the city."}}, "required": ["city"]}}}]}
```

The data is radically different but gives us the underlying structure of how the agents are thinking in ReACT terms.  If we fail to clean and convert data to Dataframes, then our initial reaction would be to implement *pd.json_normalize()* because our failures might be due to varying dict nesting depths.

### How would you prepare this data for a neural network?
If we were to feed this into say an NLP engine like a Bidirectional GRU, LSTM, or even a full-on BeRT training, then we'd want all of the data from each type of policy.  Despite each one having different language to describe each section, the structure is equivalent; therefore, moving forward, we would aim to clean the data into a dict and use the from_dict() functionality in pandas to build our frames.

When it comes to the specific language, finding unigrams and bigrams of each section, then treating them as individual features would be the best approach.  Since "escalation" language will pollute the lexicographical frequency of say an agent's introduction instruction.  In addition, would want to explore what is word frequency in the agent's thought pattern, and conclude whether the platform is using the correct language to be the most efficient.  We can sanity check our work and make sure our weights aren't diverging or gaining false-positive signals.

If we TF-IDF per section then each section will down-weight external language found in the original samples, which helps us partition even further than just training each section as separate features.

When it comes to line-by-line padding and truncation of sequences we have the issue of some lines being long, while most are short.  Therefore, we would elect the idea of using the 75th percentile of the dataset to gain the majority of lines within our designated real estate.  Statistically, the 75th percentile captures the bulk of our empirical distribution while avoiding outlier-length sequences that would introduce excessive padding overhead.  If our distribution turns out to be heavy-tailed, then we would want to apply a log function to our line lengths then take the 75th percentile.

### Where could agentic automation help in this workflow?
Our dataset was published three months ago and we have no guarantee it will be continuously updated.  However, the alternative dataset contains provisioned safetensors, which constrains work to fine-tuning frozen layers rather than the raw feature engineering this module requires. Future work should verify whether the dataset has been updated, deprecated, or superseded before extending these findings.

Data quality can be compared daily and detect schema drift in our dict generation, if the dataset is updated or heavily modified.  However, the dataset should merely be appended data onto what we already have, in which case, we would partition the data from what we have already trained and then continue training on previous weights being loaded in.  Retraining the whole dataset could help if there were any improvements made to the data or a large quantity of word refactoring.  Since we might not have this data tomorrow, next week, etc., we would maintain a versioned copy in cloud object storage with its SHA-256 hash, compared daily against the upstream source.

Our pipelines need a mandatory version control implemented so we can track schema drift and preserve our methods in case NVIDIA reverts anything, or reuses old patterns.