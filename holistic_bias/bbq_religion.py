import json

# Caminho para o arquivo
file_path = "Religion.jsonl"

# Carrega e mostra um sample de 3 linhas
with open(file_path, "r", encoding="utf-8") as f:
    samples = [json.loads(next(f)) for _ in range(3)]

for i, sample in enumerate(samples, 1):
    print(f"Sample {i}:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    print("-" * 40)