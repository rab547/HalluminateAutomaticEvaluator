import os
import time
import re
import pandas as pd
import dspy
from dspy import Example
from dotenv import load_dotenv
import tqdm

file_name = "Data/Copy of 2-28-25 - Temporal Eval.csv"
second_round = "Data/Copy of 2-13-25 Booo Eval - Temporal.csv"

def is_error_response(text): #Combine patterns 
    error_patterns = [ r"<title>Internal Server Error</title>", r"<h1>.*Internal Server Error.*</h1>" ]  
    # Check for patterns (using re.DOTALL to match across newlines) 
    for pattern in error_patterns: 
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL): return True  
    return False

load_dotenv() 
# Configure providers
groq_lm = dspy.LM(
    "groq/llama-3.2-90b-vision-preview",
    api_key=os.getenv("GROQ_API_KEY")
)
#not confiugred yet
azure_lm = dspy.LM(
    "azure/your-deployment-name", #not sure what htis is yet, but might not need it lowkey
    api_key=os.getenv("AZURE_API_KEY"),
    api_base=os.getenv("AZURE_API_BASE"),
    api_version="2023-12-01-preview"
)
google_lm = dspy.LM(
    "gemini/gemini-2.0-flash-thinking-exp-01-21",  # Use appropriate model name
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
dspy.configure(lm=groq_lm, temperature=0)

# Define Grading with Chain of Thought without few shot learning
class CoTAnswerGrader(dspy.Signature):
    """Evaluate if the LLM's answer matches the ground truth with reasoning."""
    question = dspy.InputField()
    llm_answer = dspy.InputField()
    ground_truth = dspy.InputField()
    assessment = dspy.OutputField(desc="'pass' if the LLM answer captures **all core factual details** of the ground truth **without distortions**; "
        "'fail' if it omits key information, adds incorrect details, or changes meaning.")
    reasoning = dspy.OutputField(desc=
        "1. Identify the core fundamental details in the ground truth that directly answer the question. "
        "2. Compare the LLM answer to see if it captures these key details. "
        "3. Highlight any differences and analyze their relevance (e.g., missing, incorrect, extra details). "
        "4. Explain why these differences make the LLM answer incorrect, or if they are minor enough to still allow a 'pass'."
    )

class CoTGrader(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(CoTAnswerGrader)  # Uses step-by-step reasoning

    def forward(self, question, llm_answer, ground_truth):
        return self.assess(
            question=question,
            llm_answer=llm_answer,
            ground_truth=ground_truth
        )

# Load dataset
df = pd.read_csv(file_name)
dataset = [
    Example(
        question=row["Question"],
        llm_answer=row["ctrl_panel_answer"],
        ground_truth=row["Ground Truth Answer"],
        ground_truth_label=row["Label"].lower()
    ).with_inputs("question", "llm_answer", "ground_truth")  # Mark these fields as inputs
    for _, row in df.iterrows()
]
# Initialize the CoT evaluation model
cot_grader = CoTGrader()

# Evaluate dataset
correct = 0
total = len(dataset)

for example in tqdm.tqdm(dataset, desc="Evaluating answers"):
    returnable = ""
    if is_error_response(example.llm_answer):
        returnable = "error generating"
    else:
        time.sleep(0)  # Reduce API request rate
        prediction = cot_grader(
            question=example.question,
            llm_answer=example.llm_answer,
            ground_truth=example.ground_truth
        )
        returnable = prediction.assessment.lower()
    if returnable == example.ground_truth_label.lower():
        correct += 1

# Print final score
print(f"\nGrading Accuracy: {correct}/{total} ({correct/total:.2%})")

# Display sample reasoning output
# first_example = dataset[0]
# first_prediction = cot_grader(
#     question=first_example.question,
#     llm_answer=first_example.llm_answer,
#     ground_truth=first_example.ground_truth
# )
# print("\nSample CoT Explanation:")
# print(f"Q: {first_example.question}")
# print(f"LLM Answer: {first_example.llm_answer}")
# print(f"Ground Truth: {first_example.ground_truth}")
# print(f"Assessment: {first_prediction.assessment}")
# print(f"Reasoning: {first_prediction.reasoning}")

df = pd.read_csv(second_round)
dataset = [
    Example(
        question=row["Question"],
        llm_answer=row["ctrl_panel_answer"],
        ground_truth=row["Ground Truth Answer"],
        ground_truth_label=row["Label"].lower()
    ).with_inputs("question", "llm_answer", "ground_truth")  # Mark these fields as inputs
    for _, row in df.iterrows()
]



def binary_classification_metric(example, prediction, trace=None):
    # Get prediction result (lowercased for consistency)
    predicted_label = prediction.assessment.lower()
    
    # Get ground truth label (already lowercased in your data processing)
    true_label = example.ground_truth_label
    
    # Return True if they match, False otherwise
    return predicted_label == true_label

optimizer = dspy.BootstrapFewShot(
    metric=binary_classification_metric,     # The metric function we just defined
    max_bootstrapped_demos=8,                # Maximum number of bootstrapped examples
    max_labeled_demos=8,                     # Maximum number of labeled examples
    metric_threshold=0.7,                    # Minimum performance threshold for bootstrapped examples
    max_rounds=10,                           # Number of optimization trials
    teacher_settings={"temperature": 0}    # Parameters for bootstrapping
)

optimized_grader = optimizer.compile(
    student=cot_grader,
    trainset=dataset,  # Your training examples
)


correct = 0
total = len(dataset)
time.sleep(5)
for example in tqdm.tqdm(dataset, desc="Evaluating with optimized model"):
    returnable = ""
    if is_error_response(example.llm_answer):
        returnable = "error generating"
    else:
        time.sleep(0)  # Reduce API request rate
        prediction = optimized_grader(
            question=example.question,
            llm_answer=example.llm_answer,
            ground_truth=example.ground_truth
        )
        returnable = prediction.assessment.lower()
    if returnable == example.ground_truth_label.lower():
        correct += 1

# Print final score
print(f"\nOptimized Grading Accuracy: {correct}/{total} ({correct/total:.2%})")