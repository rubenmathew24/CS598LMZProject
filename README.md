# CS598LMZProject
Project for Software QA w/ Generative AI


## How to use Scraper Files:

### scraper.py

Given a dictionary of topics (currently needs to be edited directly in file) creates normalized log files in Title/Question/Answer Format from relevant stack overflow questions.
Dictionary should be formatted as {key: value} where key is the repository id, and value is a query (likely an error statement)

Once the dictionary is created, run `python scraper.py`

### embed_for_rag.py

This file either creates or deletes the PostgreSQL Database.

Using HomeBrew:
```
brew install postgresql
brew install pgvector
brew services start postgresql
```

This will start the PostgreSQL Service. To embed and store each log file, simply run one of the following commands:

```
python embed_for_rag.py
```
This will create a new database called 'rag_embeddings'

```
python embed_for_rag.py --reset
```
This will forcibily delete any existing database called 'rag_embeddings' before creating a new instance

Once the process is done you can use:

```
psql -d rag_embeddings
SELECT * FROM documents;
```
This will return every embedding in the database.

If you wish to remove the database simply run

```
python embed_for_rag.py --clear
```
This will delete any existing database called 'rag_embeddings'

## Iterative Repair

For initial setup, localization, repair, and patch validation and selection follow the original github repository README: https://github.com/OpenAutoCoder/Agentless.git

After generating the preliminary patch locations and concatenating the reproduction results for the 10 patches per 4 edit locations into a json, the following command can be run to produce 5 new patches.

```bash
python agentless/repair/iterative_repair.py \
  --loc_file        results/swe-bench-lite/edit_location_individual/loc_merged_0-0.jsonl \
  --repro_results   results/swe-bench-lite/repair_sample_1all_repro_flags.jsonl \
  --output_folder   results/swe-bench-lite/repair_sample_iterative_1 \
  --max_samples     5 \
  --model           gpt-4o-2024-05-13 \
  --backend         openai \
  --dataset         princeton-nlp/SWE-bench_Lite
```

Additional Repair Commands:

```bash
for i in {1..3}; do
  python agentless/repair/iterative_repair.py \
    --loc_file        results/swe-bench-lite/edit_location_individual/loc_merged_${i}-{i}.jsonl \
    --repro_results   results/swe-bench-lite/repair_sample_${i+1}/all_repro_flags.jsonl \
    --output_folder   results/swe-bench-lite/repair_sample_iterative_${i+1} \
    --max_samples     5 \
    --model           gpt-4o-2024-05-13 \
    --backend         openai \
    --dataset         princeton-nlp/SWE-bench_Lite
done
```

These commands generate 5 samples each (1 greedy and 4 via temperature sampling) as defined --max_samples 5. The --context_window indicates the amount of code lines before and after each localized edit location we provide to the model for repair. The patches are saved in results/swe-bench-lite/repair_sample_iterative_{i}/output.jsonl, which contains the raw output of each sample as well as any trajectory information (e.g., number of tokens). The complete logs are also saved in results/swe-bench-lite/repair_sample_iterative_{i}/repair_logs/

The above commands will combine to generate 20 samples in total for each bug.

Follow the setup from the original Agentless github repository for the remaining steps to evaluate these patches using the regression and reproduction tests. The rerank command will generate a new preds.json, which can be test on SWE-Bench Lite. 

# RAG
Rag is set up in such a fashion as to allow the user to add arbitrary information (such as documentation) to the database and allow for it to be retrieved by the repair script when in the process of repair. The user must run the rag script `Agentless/agentless/repair/rag.py` with proper API keys for Gemini models. There are some urls there as a place holder, but the user can edit this to add more by changing the `target_urls` variable. Note, that due to how Gemini API is set up, it may run into rate limiting issues. 
