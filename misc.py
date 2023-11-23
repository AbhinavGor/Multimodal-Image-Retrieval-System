import pymongo

# Assuming you have a MongoDB client and a database
client = pymongo.MongoClient("mongodb://localhost:27017/")
database = client["cse515_project_phase1"]
collection = database["features"]

# Update all documents, converting image_id to integer
collection.update_many({}, {"$set": {"image_id": {"$toInt": "$image_id"}}})

# Print a message indicating the update is complete
print("All documents updated successfully.")
