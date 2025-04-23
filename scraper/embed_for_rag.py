from google import genai
from google.genai import types
import os
from tqdm import tqdm
from termcolor import colored
import psycopg2
import argparse

# Get the API key from the environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_LENGTH = 3072

# Embed text with Gemini for retrieval
def embed(text:str):

	client = genai.Client(api_key=GEMINI_API_KEY)
	result = client.models.embed_content(
			model="gemini-embedding-exp-03-07",
			contents=text,
			config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
	)
	return result.embeddings[0].values


# Initialize the database for storing embeddings
def initialize_database(
	server_dbname="postgres",
	new_dbname="rag_embeddings",
	host="localhost",
	port="5432",
	user=os.getenv("LOGNAME"),
	embedding_dim=EMBEDDING_LENGTH
):

	# Step 1: Connect to the Postgres server to create a new database
	try:
		conn = psycopg2.connect(
			dbname=server_dbname,
			user=user,
			host=host,
			port=port
		)
		conn.autocommit = True
		cur = conn.cursor()
	except psycopg2.OperationalError:
		print(colored("Likely haven't installed postgresql, or haven't started the service", color="red", attrs=["bold"]))
		print("\tUse Homebrew to install if you haven't:  'brew install postgresql'")
		print("\tStart the service after it is installed: 'brew services start postgresql'")
      
	# Create the new database
	try:
		cur.execute(f"CREATE DATABASE {new_dbname};")
		print(f"Database '{new_dbname}' created.")
	except psycopg2.errors.DuplicateDatabase:
		print(f"Database '{new_dbname}' already exists.")

		# Assume if database exists, properly initialized
		print(colored("Try using --reset if you want to clear the existing Database before initialization", color="yellow"))
		return

	cur.close()
	conn.close()
      
	# Step 2: Connect to the new database
	conn = psycopg2.connect(
		dbname=new_dbname,
		user=user,
		host=host,
		port=port
	)
	cur = conn.cursor()

	# Step 3: Create the pgvector extension if needed
	try:
		cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
		conn.commit()
		print("pgvector extension is ready.")
	except psycopg2.errors.UndefinedFile:
		print(colored("Likely haven't installed pgvector", color="red", attrs=["bold"]))
		print("\tUse Homebrew to install if you haven't:                                     'brew install pgvector'")
		print("\tIf you installed, restart the serice to make sure changes went into effect: 'brew services restart postgresql@14'")

	# Step 4: Create the documents table
	cur.execute(f"""
		CREATE TABLE IF NOT EXISTS documents (
			id SERIAL PRIMARY KEY,
			repo_id TEXT,
			embedding VECTOR({embedding_dim})
		);
	""")
	conn.commit()
	print("Table 'documents' is ready.")

	cur.close()
	conn.close()
	print("Database initialization complete.")

# Delete Database for clean testing
def delete_database(
	server_dbname="postgres",
    target_dbname="rag_embeddings",
    host="localhost",
    port="5432",
    user=os.getenv("LOGNAME")
):
    conn = psycopg2.connect(
        dbname=server_dbname,
        user=user,
        host=host,
        port=port
    )
    conn.autocommit = True
    cur = conn.cursor()

	# Drop
    try:
        # Terminate all active connections to the target database
        cur.execute(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s
            AND pid <> pg_backend_pid();
        """, (target_dbname,))
        
        # Now drop the database
        cur.execute(f"DROP DATABASE IF EXISTS {target_dbname};")
        print(f"Database '{target_dbname}' deleted.")
    except Exception as e:
        print(f"Error deleting database: {e}")
    finally:
        cur.close()
        conn.close()

# Inserts the embeddings to the psql db
def insert_embedding(
	repo_id: str,
	embedding: list,
	dbname="rag_embeddings",
	host="localhost",
	port="5432",
	user=os.getenv("LOGNAME")
	):

	if len(embedding) != EMBEDDING_LENGTH:
		print(colored("Not the Embedding Length Expected...", color="yellow"))
		
	try:
		conn = psycopg2.connect(
			dbname=dbname,
			user=user,
			host=host,
			port=port
		)
		cur = conn.cursor()

		cur.execute(
			"INSERT INTO documents (repo_id, embedding) VALUES (%s, %s);",
			(repo_id, embedding)
		)

		conn.commit()
		if cur:
			cur.close()
		if conn:
			conn.close()

	except Exception as e:
		print(f"\nError inserting embedding for repo_id '{repo_id}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage the RAG embeddings database.")
    parser.add_argument(
        "--clear", action="store_true", help="Delete the database only."
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete and then re-initialize the database."
    )
    args = parser.parse_args()

    if args.clear:
        delete_database()
    elif args.reset:
        delete_database()
        initialize_database()
    else:
        initialize_database()

if __name__ == "__main__":
	
	main()

	# Grab each file in test_logs
	log_folders = os.listdir("test_logs")

	for folder in tqdm(log_folders, desc="Repositories Completed", colour="green", total=len(log_folders)):
		folder_path = f"test_logs/{folder}"
		files = os.listdir(folder_path)

		for file in tqdm(files, desc="Embeddings Completed", colour="green", total=len(files), leave=False):
			# Read log
			with open(f"{folder_path}/{file}", "r") as f:
				log = f.read()
				embedding = embed(log)
				insert_embedding(folder,embedding)


