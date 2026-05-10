import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import joblib


# ----------------------------------------------------------------------------------------------------------------------
# Code Content:
# 1. Loading the data and adding Gene Symbol and Metadata
#    1.1. Download file
#    1.2. Arranging the Metadata
#    1.3. Arranging The RNA-seq data (for LUSC and LUAD)
#    1.4. Gene Annotation
#    1.5. Preparing X and y

# * This part already done in project 1 ( as a validation data)
# ----------------------------------------------------------------------------------------------------------------------




# ----------------------------------------------------------------------------------------------------------------------
# 1Loading the data and adding Gene Symbol and Meta data--------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# The Training Data:
# The TCGA–GTEx (TOIL recompute) dataset is a large-scale RNA-seq compendium combining tumor samples from TCGA with
# normal tissue samples from the GTEx project, all processed through a unified TOIL pipeline. In this study, only
# lung-related samples were used, including lung cancer tumors from TCGA (LUAD and LUSC) and normal lung tissue from
# GTEx. Gene expression is provided as normalized TPM values with gene symbols as features and samples as columns.
# This dataset served as an external validation cohort, enabling assessment of model generalization across independent
# sources while accounting for potential batch effects between TCGA and GTEx.TCGA


# 1.1. Download file ---------------------------------------------------------------------------------------------------
# https://xenabrowser.net/datapages/?dataset=TcgaTargetGtex_rsem_gene_tpm&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
# I downloaded three files:
# 1. Phenotype: https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz; Full metadata
# 2. Data: TCGA TARGET GTEx : https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz; Full metadata
# 3. Annotation: https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fgencode.v23.annotation.gene.probemap; Full metadata


# 1.2. Arranging the Metadata ------------------------------------------------------------------------------------------
# I am starting from the metadata becuse the data contains many cancer type I don't need and I want to know which to filter
pheno_path = r"G:\My Drive\Shimon\stuffs\computational_biology_Free_lance_Project\Self_projects\Project_1_Cancer_classification\data\External Data For validation\TCGA-GTEx Lung (TOIL)\TcgaTargetGTEX_phenotype.txt"
pheno_TCGA_GTEx = pd.read_csv( pheno_path, sep="\t", encoding="latin1")
pheno_TCGA_GTEx.columns
pheno_TCGA_GTEx[pheno_TCGA_GTEx['_primary_site'].str.contains('Lung', na=False)]['_sample_type'].unique()

#  keeping only Lung cancer metadata
# Keeping only the Lung cancer
pheno_TCGA = pheno_TCGA_GTEx[
    (pheno_TCGA_GTEx['_primary_site'] == 'Lung') &
    (pheno_TCGA_GTEx['_sample_type'].isin([
        'Primary Tumor',
    ]))].copy()

# Adding label
pheno_TCGA['label'] = pheno_TCGA['detailed_category'].map({
    'Lung Squamous Cell Carcinoma': 1,
    'Lung Adenocarcinoma': 0})

# Sample count:
print(pheno_TCGA['detailed_category'].value_counts())
print(pheno_TCGA['label'].value_counts())
sample_ids = pheno_TCGA['sample'].values
len(sample_ids)

cols = pd.read_csv(  #get column names
    r"G:\My Drive\Shimon\stuffs\computational_biology_Free_lance_Project\Self_projects\Project_1_Cancer_classification\data\External Data For validation\TCGA-GTEx Lung (TOIL)\TcgaTargetGtex_rsem_gene_tpm.gz",
    sep="\t",
    nrows=0)


#  1.3. Arranging The RNA-seq data (for LUSC and LUAD) -----------------------------------------------------------------
cols_to_use = ['sample'] + [c for c in cols if c in sample_ids]
data = pd.read_csv(  #get the data ( only for Lung cancer)
    r"G:\My Drive\Shimon\stuffs\computational_biology_Free_lance_Project\Self_projects\Project_1_Cancer_classification\data\External Data For validation\TCGA-GTEx Lung (TOIL)\TcgaTargetGtex_rsem_gene_tpm.gz",
    sep='\t',
    usecols=cols_to_use)
print(data)
data = data.set_index('sample')


# 1.4. Gene Annotation -------------------------------------------------------------------------------------------------
data.index = data.index.str.split('.').str[0]  # remove version suffix
annot = pd.read_csv(r"G:\My Drive\Shimon\stuffs\computational_biology_Free_lance_Project\Self_projects\Project_1_Cancer_classification\data\External Data For validation\TCGA-GTEx Lung (TOIL)\probeMap_gencode.v23.annotation.gene.probemap",
                    sep="\t")
annot_clean = annot[['id', 'gene']].copy()
annot_clean = annot_clean.dropna()
annot_clean = annot_clean[annot_clean['gene'] != '']
annot_clean['id'] = annot_clean['id'].str.split('.').str[0]
data_merge = data.merge(annot_clean, left_index=True, right_on='id', how='inner')
data_merge =data_merge.drop('id', axis=1)
data_merge = data_merge.set_index('gene')
data_merge = data_merge.groupby(data_merge.index).mean()

# 1.5. Preparing X and y -----------------------------------------------------------------------------------------------
X = data_merge.T
y = pheno_TCGA.set_index('sample').loc[X.index, 'label']
print(X.shape)
print(y.shape)
print(y.value_counts())
print((X.index == y.index).all())