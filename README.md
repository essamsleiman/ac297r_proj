# mimic-pipeline-tools/mimic-iv-v2-0/
Pipeline tools for people working with mimic-iv datasets.

Created/Maintained by Jiayu (@jiayu on slack) for now


#### Repo structure
*  mimic-playgroud.ipynb -- demonstration of how to use this pipeline
*  pipeline_utils.py -- helper functions of mimic pipeline
*  datapackage_io_util.py -- helper functions to read csv
*  resources/ -- include files such as variable range / item groupers
*  mimic_hypotension_cut.ipynb -- this is used to extract patients with hypotension; need to re-adapt to new MIMIC-IV pipeline ***(TBD)***


#### Instructions
1. Complete the CITI training (Contack finale. There's detailed instructions in Google drive dtak-basics folder ***MIMIC Infrastructure***)
2. Get Access to Odyssey (Contack finale. There's detailed instructions in Google drive dtak-basics folder ***FAS Infrastructure***)
3. Once finish CITI training (it is important NOT to skip step 1 & 2), you can download the data to your local folder by
   > scp -r "username"@login.rc.fas.harvard.edu:/n/holylabs/LABS/doshi-velez_lab/Lab/mimic_iv_2.0/query_data "local_path"
4. clone this repo
  1. > git clone git@github.com:dtak/mimic-pipeline-tools.git
  2. > cd cd mimic-pipeline-tools/mimic-iv-v2-0/
5. open **mimic-playgroud.ipynb**
  1. replace  **data_path** to where the MIMIC-IV query data is
  2. replace resource_path 
     > **repo_path**/mimic-pipeline-tools/mimic-iv-v2-0/resources/
  3. For instructions of how to use **mimic-playgroud.ipynb**, please read the notebook



#### MIMIC-IV misc note (difference between MIMIC-III and MIMIC-IV):
1. MIMIC-IV uses stay_id instead of icustay_id
2. There is no structured documentation of admission diagnosis in MIMIC-IV
3. ethinicity ---> race
4. MIMIC-IV only has data from MetaVision
5. no imputation needed for static info (demo + sepsis +commorb)
6. MIMIC-IV should not have nan valuenums for vitals and labs
7. MIMIC-IV labevents table has reference range for normal lab value but it is not consistent across rows so I decide to exlude it for now


