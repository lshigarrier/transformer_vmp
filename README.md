# transformer_vmp
## To train a deterministic transformer on PeMS-SF:

Create a directory `data/` at the root of the project.  
Download the [PeMS-SF dataset](http://www.timeseriesclassification.com/description.php?Dataset=PEMS-SF) and unzip it in `data/`.  
Create a directory and a subdirectory `models/pems_base1` at the root of the project.  
Then launch the training with `python -u train.py --config pems_base`