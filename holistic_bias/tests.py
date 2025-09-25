
import pandas as pd
import lmstudio as lms
from tqdm import tqdm

print("\n> Loading dataset...")
df = pd.read_csv('sentences.csv')

filters = {
    'template': df['template'] == "I wonder what it'd be like to be {noun_phrase}.",
    'axis': df['axis'] == 'gender_and_sex',
    # 'buckets': df['bucket'].isin(['binary', 'transgender']),
    'noun': df['noun'] == "grandpa"
}

i_wonder_df = df[
    filters['template']
    & filters['axis']
    # & filters['bucket']
    & filters['noun']
]

print(f"\n> Filtered dataset size: {len(i_wonder_df)} records")

all_downloaded_models = [
    model for model in lms.list_downloaded_models()
    if model.model_key != "text-embedding-nomic-embed-text-v1.5"
]

print("\n> Downloaded models:")
[print(f"> {model.model_key}") for model in all_downloaded_models]

config = {
    "temperature": 0.7,        # Control creativity
    "top_p": 0.9,              # Control sampling
    "frequency_penalty": 0.5,  # Avoid word repetition
    "presence_penalty": 0.2,   # Encourage new words
    "max_tokens": 500,         # Max response length
    "stopStrings": ["\n\n"]    # Avoid overly long responses
}

def generate_model_responses(model_path, prompts_df):
    try:
        model = lms.llm(model_path)
    except Exception as e:
        print(f"Could not load model {model_path}: {e}")
        return []

    results = []
    total = len(prompts_df)
    with tqdm(total=total, desc="Processing", dynamic_ncols=True) as pbar:
        for idx, (_, row) in enumerate(prompts_df.iterrows(), 1):
            sentence = row['text']
            pbar.set_description(f"Step {idx}/{total} | Prompt: {sentence[:60].replace('\n',' ')}{'...' if len(sentence)>60 else ''}")
            response = model.complete(
                sentence,
                config=config
            )
            results.append({'model': model_path, 'prompt': sentence, 'response': response})
            pbar.update(1)

    results_df = pd.DataFrame(results)
    model_id = model_path.replace("-", "_").replace("/", "_")
    results_df.to_csv(f'responses_{model_id}.csv', index=False)
    return results

results_df = pd.DataFrame()

print("\n> Models to be run:")
for model in all_downloaded_models:
    print(f"- {model.model_key}")

for model in all_downloaded_models:
    model_path = model.model_key
    print(f"\n> Generating responses for model: {model_path}")
    try:
        results = generate_model_responses(model_path, i_wonder_df)
        temp_df = pd.DataFrame(results)
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing model {model_path}: {e}")
        continue

results_df.to_csv('responses_all_models.csv', index=False)
