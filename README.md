# ProstaNet

## ProstaNet data processing
- To generate proteins' variants, use Rosetta relax. For more detailed steps see the Rosetta section.
- Using ```ProstaNet\Data_processing\prot2json.py``` to obtain a json file for the data input to the model. The ```prot2json.py``` will produce fasta files for all the proteins. The fasta files can be used as the input to generate the PSSM features. The psi-blast will take a long time to perform BLAST search. Our results showed that PSSM feature does not 
