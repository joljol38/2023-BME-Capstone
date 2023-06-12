# Drug Side Effect Prediction Model (2023 HUFS BME Capstone Project)

## Data
:pill: **Drug** :pill:
1. drug_side_freq.pkl <br/>
    * Drug-side effect frequency pairs obtained from SIDER and OffSIDES database. <br/>
    * This matrix has 531 drugs(rows) and 655 side-effects(columns).
2. drug_structure_similarity.pkl <br/>
    * Structure similarity of 531 drugs
    * This matrix has 531 rows and 531 columns
3. drug_gene_target.pkl <br/>
    * The target gene information of the drugs is obtained from DrugBank database (https://go.drugbank.com/).
    * This matrix has 531 drugs(rows) and 632 target genes (columns).
4. SMD_combined.pkl <br/>
    * This matrix is created by extracting the 'combined' columns from the file 'Chemical_chemical.links.detailed.v5.0.tsv.gz' in the STITCH database (http://stitch.embl.de/).
    * The range of values is from 0 to 1.
5. SMD_database.pkl
    * This matrix is created by extracting the 'database' columns from the file 'Chemical_chemical.links.detailed.v5.0.tsv.gz' in the STITCH database (http://stitch.embl.de/).
    * The range of values is from 0 to 1.
6. SMD_experimental.pkl
    * This matrix is created by extracting the 'experimental' columns from the file 'Chemical_chemical.links.detailed.v5.0.tsv.gz' in the STITCH database (http://stitch.embl.de/).
    * The range of values is from 0 to 1.
7. SMD_similarity.pkl
    * This matrix is created by extracting the 'similarity' columns from the file 'Chemical_chemical.links.detailed.v5.0.tsv.gz' in the STITCH database (http://stitch.embl.de/).
    * The range of values is from 0 to 1.
8. SMD_textmining.pkl
    * This matrix is created by extracting the 'textmining' columns from the file 'Chemical_chemical.links.detailed.v5.0.tsv.gz' in the STITCH database (http://stitch.embl.de/).
    * The range of values is from 0 to 1.

:dizzy_face: **Side Effect** :dizzy_face:
1. new_glove_wordEmbedding.pkl
    * The word embedding matrix of 655 side effects. 
    * Use the 300-dimensional Global Vectors (GloVe) trained on the Wikipedia dataset to represent the information of side effects. 
    * Each row of the matrix represents the word vector encoding of a side effect.

2. side_effect_semantic.pkl
    * The semantic similarity matrix of side effects. 
    * Download side effect descriptors from Adverse Drug Reaction Classification System (ADReCS, http://bioinf.xmu.edu.cn/ADReCS/index.jsp) 
    * Each row of the matrix represents the similarity value between a side effect and all side effects in the benchmark dataset. 
    * The range of values is from 0 to 1.




## Reference

### Predicting the frequencies of drug side effects
https://www.nature.com/articles/s41467-020-18305-y

### A similarity-based deep learning approach for determining the frequencies of drug side effects
https://academic.oup.com/bib/article/23/1/bbab449/6412393?


