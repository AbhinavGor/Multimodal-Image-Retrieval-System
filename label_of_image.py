from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
coll = db.features

print(coll.find_one({"image_id": str(input("Enter image ID: "))})["target"])
