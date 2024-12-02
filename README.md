# ProstaNet

## ProstaNet data processing
- To generate proteins' variants, use Rosetta relax. For more detailed steps see the Rosetta section.
- Use ```ProstaNet\Data_processing\prot2json.py``` to generate a JSON file for the data input to the model. The ```prot2json.py``` script will produce FASTA files for all the proteins. These FASTA files can be used as input to generate the PSSM features.

  Note: Performing the PSI-BLAST search may take a long time. However, our results show that omitting the PSSM features does not significantly affect the outcome. Therefore, if you do not wish to wait for the BLAST search to complete, you may skip this step.

  For detailed instructions on the BLAST search process, refer to the PSI-BLAST section.
- Utilize ```ProstaNet\Data_processing\json2graph.py``` to convert the JSON files to their corresponding protein graph.

## Train ProstaNet model
- Run ```ProstaNet\Model\Train_GVP.py``` to train the model. The default training set for single-point mutations is ```Datasets\Single_training.npy```, for multiple-mutation is ```Datasets\Multiple_training_cluster.npy```.

  You can use your own training set and modify the parameters in the model.

## Test ProstaNet model
- Run ```ProstaNet\Model\Test.py``` to test the train models. The default testing set for single-pint mutations is 
