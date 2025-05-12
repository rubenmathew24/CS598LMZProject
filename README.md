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
