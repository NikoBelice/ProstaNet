# ProstaNet

## Install python dependencies
```
conda env create --file environment.yml
```

## ProstaNet data processing
- To generate proteins' variants, use Rosetta code. The wild-type proteins need to be relaxed. For more detailed steps see the Rosetta section.
- Use ```ProstaNet\Data_processing\prot2json.py``` to generate a JSON file for the data input to the model. The ```prot2json.py``` script will produce FASTA files for all the proteins. These FASTA files can be used as input to generate the PSSM features.

  Note: Performing the PSI-BLAST search may take a long time. However, our results show that omitting the PSSM features does not significantly affect the outcome. Therefore, if you do not wish to wait for the BLAST search to complete, you may skip this step.

  For detailed instructions on the BLAST search process, refer to the PSI-BLAST section.
- Utilize ```ProstaNet\Data_processing\json2graph.py``` to convert the JSON files to their corresponding protein graph.

## Use the model to make prediction
- Convert the proteins you want to predict by following the data processing section.
- Generate a data list. The format of the single-point mutations list should follow ```Single_training.npy```, multiple-point mutations list follows ```Multiple_training_cluster.npy```.
- Run ```ProstaNet\Model\Predict.py```.

## Train/fine-tune ProstaNet model
- Run ```ProstaNet\Model\Train_GVP.py``` to train the model. The default training set for single-point mutations is ```Single_training.npy```, for multiple-mutations is ```Multiple_training_cluster.npy```.
- For fine-tune pre-trained model, run ```ProstaNet\Model\Train_GVP_finetune.py```

  You can use your own training set and modify the parameters in the model.

## Test ProstaNet model
- Run ```ProstaNet\Model\Test.py``` to test the train models. The default testing set for single-pint mutations is ```Ssym_testing.npy``` and ```Extra_testing.npy```, for multiple-mutations is ```Multiple_testing_cluster.npy```.

## Protein relax and mutate
- Use the Rosetta command in ```ProstaNet\Data_processing\rosetta_mutate.py``` and ```rosetta_mutate_multiple.py``` to generate protein variants. Wild-type protein uses ```rosetta_relax.py``` to relax.

## PSI-BLAST
- Download UniRef90 database from https://www.uniprot.org/help/downloads
- Run ```ProstaNet\Data_processing\psiblast.py```

## Example
- You can use the data in ```Data_example``` and run ```Train_GVP.py``` to go through data processing and training process

### Database available
FireProtDB: https://loschmidt.chemi.muni.cz/fireprotdb/

ThermoMutDB: https://biosig.lab.uq.edu.au/thermomutdb/

ProThermDB: https://web.iitm.ac.in/bioinfo2/prothermdb/

MPTherm: https://www.iitm.ac.in/bioinfo/mptherm/

S2648: https://protddg-bench.github.io/s2648/
