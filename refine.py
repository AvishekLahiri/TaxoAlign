import pickle
from tqdm import tqdm

from openai import OpenAI
from prompt import refine_taxonomy
import pandas as pd
import re

def extract_input_and_prediction(text):
    # Extract input text and prediction from the given text
    input_pattern = r"Input:\s+(.*?)\s+###"  # Regex pattern to find the input text
    prediction_pattern = r"Response:\s+(.*)$"  # Regex pattern to find the prediction text
    try:
        input_text = re.search(input_pattern, text, re.DOTALL).group(1).strip()  # Extract input text
        prediction = re.search(prediction_pattern, text, re.DOTALL).group(1).strip()  # Extract prediction text
        #print("Input Text = ", input_text)
    except:
        prediction = ""

    #print("Prediction = ", prediction)
    input_text = input_text.replace('-', ' ')  # Replace hyphens with spaces in input text
    prediction = prediction.replace('-', ' ')  # Replace hyphens with spaces in prediction text
    return input_text, prediction


def get_model_response(client, prompt, seed):

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a researcher who can summarize research documents"},
            {
                "role": "user",
                "content": prompt
            }
        ],
      temperature=0.6,
      max_tokens=1024,
      seed=seed
    )

    response = completion.choices[0].message.content
    #print(response)

    return response


if __name__ == "__main__":
    seed = 42
    client = OpenAI(api_key = '')
    
    # Read the content of the text file
    with open('file_path', 'r') as file:
        file_content = ''.join(file.readlines())

    with open('initial_selected_papers_title_new_conf.pkl', 'rb') as f:
        initial_selected_papers_title = pickle.load(f)

    file_content = file_content.replace("Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.", "99999Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.")

    # Split the content into individual instances based on the marker
    instances = file_content.split("99999")

    # Remove any empty strings
    # instances = [instance.split('### Response:')[-1] for instance in instances]
    instances = [instance.strip() for instance in instances if instance.strip()]

    # Create a DataFrame to store the instances
    predicted_df = pd.DataFrame({'text': instances})

    for index, row in predicted_df.iterrows():
        input_txt, prediction = extract_input_and_prediction(row['text'])
        prediction = prediction.replace('\t', '\n')
        input_txt = input_txt.replace('\t', '\n')
        txt = input_txt.replace("Taxonomy Topic:\n", "")
        topic = txt[:txt.find('\n')].strip().lower()
        fname = ""
        for i in initial_selected_papers_title:
            t = initial_selected_papers_title[i]
            t = t.replace("\n", "")
            t = t.replace("-", " ")
            if t.replace(":", "").lower() == topic.replace(":", ""):
                fname = i
        if fname == "":
            print(topic)
        prompt = refine_taxonomy(prediction, input_txt)
        response = get_model_response(client, prompt, seed)
        f_predicted = open("save_file_path/"+str(fname)+".txt", 'w')
        f_predicted.write(response)
        f_predicted.close()


