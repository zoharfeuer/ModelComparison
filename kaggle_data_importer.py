import kaggle as kg 

def import_data(dataset_id,destination_folder):
    # Set the Kaggle API credentials (Make sure you have the kaggle.json file in the correct location)
    kg.api.authenticate()

    # Download the dataset
    kg.api.dataset_download_files(dataset_id, path=destination_folder, unzip=True)
    
def get_datasets(datasets,path):
    for dataset in datasets.values():
        import_data(dataset, path)

