echo 'Preprocessing & Merging Starts'

# Folder path of newly fetched data
path_of_fetched_data=$PWD'/FetchedData'
path_of_processed=$PWD'/ProcessedData'
path_of_main_data=$PWD'/Data'
echo 'Preprocessing'
python examples/preproces_fetched_data.py --path $path_of_fetched_data --path_to_store $path_of_processed
echo 'Merging'
python examples/merge_processdata.py --path_data_to_insert $path_of_processed --path_main_data $path_of_main_data --path_merged_data $path_of_main_data
echo 'Completed'
echo 'Cleaning'
rm -rf $path_of_fetched_data
rm -rf $path_of_processed