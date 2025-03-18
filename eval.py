import csv
from groq import Groq
import os
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import google.generativeai as genai
from openai import AzureOpenAI
from litellm_client import generate_response
import time




#Utilized to collect all data including graded label for testing this evaluator
def parseCSVWithCorrectAnswers(file_name):
    results = []
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            column_mapping = {
                'Question': 'question',
                'Ground Truth Answer': 'ground truth',
                'ctrl_panel_answer': 'returned answer',
                'Label': 'label'
            }
            for row in csv_reader:
                filtered_row = {}
                for orig_key, new_key in column_mapping.items():
                    if orig_key in row:
                        filtered_row[new_key] = row[orig_key]
                if filtered_row:
                    results.append(filtered_row)  
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

    return results

#Utilized to collect input data and grade each query/response pair
def parseCSVWithoutCorrectAnswers(file_name):
    results = []
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            column_mapping = {
                'Question': 'question',
                'Ground Truth Answer': 'ground truth',
                'ctrl_panel_answer': 'returned answer'
            }
            for row in csv_reader:
                filtered_row = {}
                for orig_key, new_key in column_mapping.items():
                    if orig_key in row:
                        filtered_row[new_key] = row[orig_key]
                if filtered_row:
                    results.append(filtered_row)

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

    return results

#Utilized to take in the input data and assign each query/response pair a PASS/FAIL/ERROR label and give reasoning
def evaluateAnswers(input):
    load_dotenv()
    #GROQ
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    #GEMINI
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
    # model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

    #OPEN AI AZURE
    # client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # api_version="2025-02-01-preview")


    for row in tqdm(input, desc="Grading responses"):
        # time.sleep(1.5)
        prompt ="""
You are an expert evaluator for question-answering systems. Your task is to assess whether a generated answer correctly addresses a question by comparing it to a known ground truth answer.

Instructions: Carefully analyze the question, ground truth correct answer, and the system's returned answer provided below. Then, assign one of the following labels:

ERROR: The returned answer contains an error stack trace or system error message instead of an actual answer.
PASS: The returned answer successfully captures all core aspects of the ground truth, even if phrased completely differently. All key information and critical details are present. Additional information is okay.
FAIL: The returned answer is missing key aspects of the ground truth, contains incorrect information, or provides misleading details. This includes partial answers that omit critical information. 
After assigning the label, provide a concise explanation (1 sentences) justifying your decision. Highlight specific elements from both the ground truth and returned answer that influenced your evaluation.

Content to Evaluate:
Question: """ + row["question"] + """
Ground Truth Correct Answer: """ + row['ground truth'] + """
Returned Answer: """+ row['returned answer'] + """
Your Evaluation
Based on the above content, provide in the following format:

Assigned_Label: [PASS/FAIL/ERROR]
Returned_Explanation: [Your justification for the assigned label]
"""


        #GROQ
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                        "role": "user",
                        "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        llm_response = completion.choices[0].message.content

        #GEMINI 
        # response = model.generate_content(
        #     prompt,
        #     generation_config={
        #         "temperature": 0, 
        #         "max_output_tokens": 1024
        #     }
        # )
        # llm_response = ""
        # try:
        #     llm_response = response.text.strip()
        # except:
        #     llm_response = "Assigned_Label: No Response Created"

        #OPENAI
        # llm_response = generate_response(
        #         model="o1-azure",
        #         messages=[
        #         {
        #                 "role": "user",
        #                 "content": prompt
        #         }
        #     ],
        #     #only temp of 1 available of o3 mini
        #     temperature=1,
        #     max_completion_tokens=1024,
        #     stream=False,
        #     stop=None,
        #     )


        # print(llm_response)
        label = None
        explanation = None


        for line in llm_response.strip().split('\n'):
            if line.startswith("Assigned_Label:"):
                label_part = line.split(":", 1)[1].strip()
                # Extract label from [PASS/FAIL/ERROR] format
                if "PASS" in label_part:
                    label = "PASS"
                elif "FAIL" in label_part:
                    label = "FAIL"
                elif "ERROR" in label_part:
                    label = "ERROR"

            elif line.startswith("Returned_Explanation:"):
                explanation = line.split(":", 1)[1].strip()

        row['label'] = label
        row['explanation'] = explanation

    return input

#Prints a report of how well the evaluator performed
def gradeAnswers(manuallyGraded, LLMGraded):
    count = 0
    correct = 0
    incorrect = []
    for i in tqdm(range(min(len(manuallyGraded), len(LLMGraded)))):
        manualLabel = manuallyGraded[i]["label"]
        LLMLabel = LLMGraded[i]['label']
        count += 1
        if manualLabel == LLMLabel or (manualLabel == "Error Generating" and LLMLabel == "ERROR"):
            correct += 1
        else:
            incorrect.append([i+1, manuallyGraded[i], LLMGraded[i]])

    if len(incorrect) != 0:
        print("Incorrect Evaluations: ")
        for subArray in incorrect:
            print("Incorrect Row: " + str(subArray[0]))
            print("Question: " + str(subArray[1]['question']))
            print("Correct Answer: " + str(subArray[1]['ground truth']))
            print("Returned Answer: " + str(subArray[1]['returned answer']))
            print("Correct Label: " + str(subArray[1]['label']))
            print("LLM Evaluation and Reasoning: " + str(subArray[2]['label']) + ": " + str(subArray[2]['explanation']))
            print()

    print("\nEvaluator matched manual grading " + str(correct) + " out of " + str(count) + " times, which is a " + str((correct/count)*100) + " percent success rate.\n")

def outputGradedCSV(gradings, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=gradings[0].keys())
        writer.writeheader()
        writer.writerows(gradings)

def evaluateAndOutput(filePath):
    # manuallyGraded = parseCSVWithCorrectAnswers(filePath)
    llmGraded = evaluateAnswers(parseCSVWithoutCorrectAnswers(filePath))
    # gradeAnswers(manuallyGraded, llmGraded)
    outputGradedCSV(llmGraded, "OUTPUT_" + filePath)
    print("Grading complete. Output saved to OUTPUT_" + filePath)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    filePath = " ".join(sys.argv[1:])
    evaluateAndOutput(filePath)



