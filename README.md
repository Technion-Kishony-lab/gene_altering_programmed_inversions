# Systematic identification of gene-altering programmed inversions across the bacterial domain
This is the code used to produce the results described in Milman et al., 2022.

## SETUP
Download gene_altering_programmed_inversions (this repository directory) to a Linux machine (we used Ubuntu 20.04.4), install Anaconda (https://www.anaconda.com/), and execute the following commands in a terminal:
```
cd gene_altering_programmed_inversions
conda create -n prog_inv_env --file conda_env.txt -c anaconda -c conda-forge -c bioconda
conda activate prog_inv_env
export PYTHONPATH=${PYTHONPATH}:`pwd`
python -Werror searching_for_pis/search_for_pis_unittests.py
```
A RuntimeError should be raised, asking you to edit two variables in gene_altering_programmed_inversions/generic/bio_utils.py to store your email and NCBI API key (https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/). This is required because the pipeline queries NCBI databases.
After filling these, try running again:
```
python -Werror searching_for_pis/search_for_pis_unittests.py
```
You should now see the following output (it took 67 seconds on our server):
```
.......................................
Ran 7 tests in 67s

OK
```

Running the actual code would take a lot more time, and potentially require a fair amount of RAM (our server had 250GB RAM at the time).
To run the code, you would first have to update the configuration file
searching_for_pis/massive_screening_configuration.py (which is mostly a huge python dict) to match your desired input
and output paths (you could also modify any argument to the pipeline. in general, we attempted to have no hardcoded
thresholds in the code, and virtually all thresholds should be read from the configuration file). Specifically, for the
steps in the pipeline that require a local BLAST nt database, you must change 'local_blast_nt_database_path' and
'local_blast_nt_database_update_log_file_path' (see more explanations in the configuration file near these keys), and for
the step that uses SRA data, you have to manually download SRA files, e.g., using:
```
curl -o SRR9952487.1 https://sra-downloadb.be-md.ncbi.nlm.nih.gov/sos3/sra-pub-run-19/SRR9952487/SRR9952487.1
```
and move these SRA files into the directory specified by 'output_dir_path' under 'stage6' and 'sra_entries_dir_name'
(see more explanations in the configuration file near these keys).

To run the whole pipeline:
```
python searching_for_pis/massive_screening_stage_1.py
python searching_for_pis/massive_screening_stage_2.py
python searching_for_pis/massive_screening_stage_3.py
python searching_for_pis/massive_screening_stage_4.py
python searching_for_pis/massive_screening_stage_5.py
python searching_for_pis/cds_enrichment_analysis.py
python searching_for_pis/massive_screening_stage_6.py
```
After running the pipeline, you can generate the paper figures and tables by opening
generate_paper_figures_and_tables.ipynb in JupyterLab (which is now installed if you followed the instructions in the
SETUP section), and replacing the os.chdir() statement at the top of generate_paper_figures_and_tables.ipynb so that
JupyterLab would switch to the main directory of our code (the one containing this README file). Note that your results
might be a slightly different, as the results described in the paper are those we got on January 2022, and whenever
you run it, there would probably be more available sequences (which might lead to extra findings, but also might lead to
missing some of what we found, in case different representative genomes would be chosen).

## main code files
- searching_for_pis/massive_screening_configuration.py - mostly a huge python dict containing all arguments for our
  pipeline.

stages 1-5 of the pipeline are described in Figure 1B:
- searching_for_pis/massive_screening_stage_1.py - Retrieval of representative genomes, including extraction of coding
  sequence annotations.
- searching_for_pis/massive_screening_stage_2.py - Identification of inverted repeats.
- searching_for_pis/massive_screening_stage_3.py - Assignment of inverted repeats to coding sequence pairs and
  discarding coding sequences containing repetitive inverted repeats. I.e., this stage ultimately gives the list of
  programmed inversion candidates.
- searching_for_pis/massive_screening_stage_4.py - Same-species genome choice.
- searching_for_pis/massive_screening_stage_5.py - Identification of intra-species variation.

The analysis in this file is described in Figures 2 and 3:
- searching_for_pis/cds_enrichment_analysis.py - Genomic architecture enrichment analysis, programmed inversion
  prediction (based on genomic architectures) and enrichment analysis of gene families associated with predicted
  programmed inversions.

The analysis in this file is described in Figure 4:
- searching_for_pis/massive_screening_stage_6.py - identification of variant coexistence in long-read sequencing data.

A JupyterLab notebook to generate the paper figures and tables:
- generate_paper_figures_and_tables.ipynb

<br><br>
For any question or comment, please contact orenmn@gmail.com

