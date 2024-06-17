import sys
print("Importing from src/components")
print(sys.path)

try:
    from src.components import data_ingestion
    data_ingestion.run()
except Exception as e:
    print(f"Failed to import or run data_ingestion from src.comp: {e}")
# Attempt to import and run the data_ingestion module
try:
    from components import data_ingestion
    data_ingestion.run()
except Exception as e:
    print(f"Failed to import or run data_ingestion: {e}")
