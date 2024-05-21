library(Seurat)
library(Matrix)


source('Supplemental_Code/Lalli_et_al_Rfunctions.R')



SFARI_run1.data <- ReadMtx(
    mtx = "GSE142078_RAW/GSM4219575_Run1_matrix.mtx.gz", 
    features = "GSE142078_RAW/GSM4219575_Run1_genes.tsv.gz",
    cells = "GSE142078_RAW/GSM4219575_Run1_barcodes.tsv.gz")
SFARI_run2.data <-  ReadMtx(
    mtx = "GSE142078_RAW/GSM4219576_Run2_matrix.mtx.gz", 
    features = "GSE142078_RAW/GSM4219576_Run2_genes.tsv.gz",
    cells = "GSE142078_RAW/GSM4219576_Run2_barcodes.tsv.gz")
SFARI_run3.data <-  ReadMtx(
    mtx = "GSE142078_RAW/GSM4219577_Run3_matrix.mtx.gz", 
    features = "GSE142078_RAW/GSM4219577_Run3_genes.tsv.gz",
    cells = "GSE142078_RAW/GSM4219577_Run3_barcodes.tsv.gz")

library(stringr)
colnames(SFARI_run1.data) <- str_replace_all(colnames(SFARI_run1.data), "-1", "")
colnames(SFARI_run2.data) <- str_replace_all(colnames(SFARI_run2.data), "-1", "")
colnames(SFARI_run3.data) <- str_replace_all(colnames(SFARI_run3.data), "-1", "")

colnames(SFARI_run1.data) <- paste(colnames(SFARI_run1.data),"-1", sep="") ## some cellbarcodes are shared across runs so add run number to cell name
colnames(SFARI_run2.data) <- paste(colnames(SFARI_run2.data),"-2", sep="")
colnames(SFARI_run3.data) <- paste(colnames(SFARI_run3.data),"-3", sep="")

SFARI <- CreateSeuratObject(SFARI_run1.data, min.cells = 50, project = "SFARI")
SFARI2 <- CreateSeuratObject(SFARI_run2.data, min.cells = 50, project = "SFARI")
SFARI3 <- CreateSeuratObject(SFARI_run3.data, min.cells = 50, project = "SFARI")

bctable <- read.csv('GSE142078_RAW/GSM4219575_Run1_Cell_Guide_Lookup.csv.gz', head=T)  # These lookup tables are provided in Supplementary Code
names(bctable) <- c("V1","V2")
bctable$V1 <- paste(bctable$V1,"-1",sep="")

bctable2 <- read.csv('GSE142078_RAW/GSM4219576_Run2_Cell_Guide_Lookup.csv.gz', head=T)
names(bctable2) <- c("V1","V2")
bctable2$V1 <- paste(bctable2$V1,"-2",sep="")

bctable3 <- read.csv('GSE142078_RAW/GSM4219577_Run3_Cell_Guide_Lookup.csv.gz', head=T)
names(bctable3) <- c("V1","V2")
bctable3$V1 <- paste(bctable3$V1,"-3",sep="")
all_singlets <- rbind(bctable, bctable2[,1:2], bctable3[,1:2])

## Although neurons are post-mitotic, added cell-cycle regression for generalizability 
cc.genes <- readLines('Supplemental_Code/regev_lab_cell_cycle_genes.txt')
s.genes <- cc.genes[1:43]
g2m.genes <- cc.genes[44:97]
# SFARI <- cc(SFARI, s.genes, g2m.genes)
# SFARI2 <- cc(SFARI2, s.genes, g2m.genes)
# SFARI3 <- cc(SFARI3, s.genes, g2m.genes)

SFARI_ALL <- merge(SFARI, y = c(SFARI2, SFARI3), project = "SFARI")
# SFARI_ALL <- cc(SFARI_ALL, s.genes, g2m.genes)
SFARI_ALL <- add_sgRNA(SFARI_ALL, all_singlets)


batch <- rep("None", length(names(SFARI_ALL$orig.ident)))
batch[grep("-1", names(SFARI_ALL$orig.ident))] <- "1"
batch[grep("-2", names(SFARI_ALL$orig.ident))] <- "2"
batch[grep("-3", names(SFARI_ALL$orig.ident))] <- "3"
SFARI_ALL$batch <- batch 
saveRDS(SFARI_ALL, file = "SFARI_ALL.rds")


# subset of cells that have a gRNA
SFARI_ALL_with_guide <- subset(x = SFARI_ALL, subset = gene_level_numeric > 1)

SFARI_ALL_with_guide <- mito_qc(SFARI_ALL_with_guide)
VlnPlot(object = SFARI_ALL_with_guide, features = c("nCount_RNA", "nFeature_RNA", "percent.mito"), group.by = "batch")  
SFARI_ALL_with_guide <- QC_Filter_Seurat(SFARI_ALL_with_guide)
SFARI_ALL_with_guide
saveRDS(SFARI_ALL_with_guide, file = "SFARI_ALL_with_guide.rds")

SFARI_ALL_with_guide <- PCA_TSNE(SFARI_ALL_with_guide)


## Fixing typos
library(tidyr)
library(plyr); library(dplyr)
SFARI_ALL_with_guide$gene_level <- relevel(factor(SFARI_ALL_with_guide$gene_level), "Nontargeting")
SFARI_ALL_with_guide$sgRNA <- revalue(SFARI_ALL_with_guide$sgRNA, c("PTEN_G2_" = "PTEN_G2"))
SFARI_ALL_with_guide$sgRNA <- revalue(SFARI_ALL_with_guide$sgRNA, c("ARID1B_g1" = "ARID1B_G2"))



### For paper, use SAFRI_ALL_with_guide, UMAP = 6 PCA dims, PCA resolution = 0.25.  Res = 1 to show cluster enrichment, res 0.25 to show few clusters ... Supplemental UMAP Figures
Idents(SFARI_ALL_with_guide) <- SFARI_ALL_with_guide$RNA_snn_res.0.25

library(viridis)
myPal <- magma(10)[c(2,3,4,6,7,9,10)]  ### main color palette throughout 
myPal2 <- magma(10)[c(2,3,4,6,7,9,10)]
a <- sapply(split(SFARI_ALL_with_guide$RNA_snn_res.0.25, SFARI_ALL_with_guide$gene_level), function(x){table(x)/sum(table(x))})
plot(hclust(dist(t(a))))  ### sort the columns to make the figure prettier 
b<- hclust(dist(t(a)), method="ward.D2")
barplot(a[,rev(b$order)], col=myPal)

## Removed ribosomal genes for differential gene expression analysis due to high variability across cells 
ribo_gene.indexes1 <- grep(pattern = "^RPL|^RPS", rownames(SFARI_ALL_with_guide), ignore.case=TRUE) 
SFARI_ALL_with_guide <- SFARI_ALL_with_guide[-ribo_gene.indexes1]

### Visualization of sgRNA efficiency using Seurat DotPlot 
library(ggplot2)
geneSet <- c("SETD5","PTEN","POGZ","MYT1L","MECP2","HDAC5","DYRK1A","CTNND2","CHD8","CHD2","ASH1L","ARID1B","ADNP")
Idents(SFARI_ALL_with_guide) <- SFARI_ALL_with_guide$gene_level
b <- DotPlot(SFARI_ALL_with_guide, features = geneSet, dot.scale=20, scale.by='size') + scale_color_viridis(option="magma")
b$data$id <- factor(b$data$id, rev(sort(levels(b$data$id))))
b$data$id <- relevel(b$data$id, "Nontargeting")
b + coord_flip() + scale_x_discrete(limits = rev(sort(levels(b$data$id)))) + scale_y_discrete(limits = sort(levels(b$data$id)))  ## X = sgRNA, Y = Gene Expression of indicated Gene -- basically a heatmap of candidate genes in cells grouped by sgRNA. Diagonal represents the gene expression of 'On-target' gene.  Size and colors are meaningful within rows. 




## import Seurat batch corrected dataset into Monocle
# BiocManager::install("monocle")
library(monocle)
SFARI_all <- newCellDataSet(as.matrix(SFARI_ALL_with_guide$RNA@scale.data), lowerDetectionLimit = 0.5, expressionFamily = VGAM::negbinomial.size())
SFARI_all <- detect_expression(SFARI_all)

some_mon <- SFARI_all
expressed_genes <- row.names(subset(fData(some_mon), num_cells_expressed >= 60))
some_mon <- some_mon[expressed_genes]
some_mon <- some_mon[,pData(some_mon)$num_genes_expressed > 500 & pData(some_mon)$num_genes_expressed < 5000]
some_mon <- estimateSizeFactors(some_mon)
SFARI_all <- some_mon

SFARI_all <- add_sgRNA_monocle(SFARI_all, all_singlets)  ## Add identified gRNA to cells 
SFARI_all_mon <- SFARI_all[,pData(SFARI_all)$gene_level > 0]   ### SFARI_with_guide 
#ordering_genes <- VariableFeatures(object = SFARI_ALL_with_guide)  ##  Used genes from bulk-time-course time-point-DE-genes but all pseduotime orderings based on highly variable genes gave broadly similar results because the variable genes ARE the neuronal differentiation genes 

temporal_genes <- read.table('Supplemental_Code/Temporal_gene_list_from_bulk.txt')  ## These genes are differentially expressed across timepoints in bulk RNA-seq 
ordering_genes <- as.character(temporal_genes$V1)
SFARI_all_mon <- setOrderingFilter(SFARI_all_mon, ordering_genes)
SFARI_all_mon_pt2 <- reduceDimension(SFARI_all_mon, max_components = 2, norm_method="none")

## Next command takes 30 minutes to 1 hour, and eats all my RAM
SFARI_all_mon_pt3 <- orderCells(SFARI_all_mon_pt2)

## Re-number PT states from left to right 
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 2)] <- rep(3, length(which(SFARI_all_mon_pt3$State == 2)))
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 7)] <- rep(2, length(which(SFARI_all_mon_pt3$State == 7)))
levels(SFARI_all_mon_pt3$State) <- c(levels(SFARI_all_mon_pt3$State),10) ## add dummy level for swap
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 6)] <- rep(10, length(which(SFARI_all_mon_pt3$State == 6)))
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 5)] <- rep(6, length(which(SFARI_all_mon_pt3$State == 5)))
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 10)] <- rep(5, length(which(SFARI_all_mon_pt3$State == 10)))
SFARI_all_mon_pt3$State <- droplevels(SFARI_all_mon_pt3$State) ### drop empty levels 



SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 6)] <- rep(5, length(which(SFARI_all_mon_pt3$State == 6)))
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 7)] <- rep(6, length(which(SFARI_all_mon_pt3$State == 7)))
levels(SFARI_all_mon_pt3$State) <- c(levels(SFARI_all_mon_pt3$State),10) ## add dummy level for swap
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 4)] <- rep(10, length(which(SFARI_all_mon_pt3$State == 4)))
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 5)] <- rep(4, length(which(SFARI_all_mon_pt3$State == 5)))
SFARI_all_mon_pt3$State[which(SFARI_all_mon_pt3$State == 10)] <- rep(5, length(which(SFARI_all_mon_pt3$State == 10)))
SFARI_all_mon_pt3$State <- droplevels(SFARI_all_mon_pt3$State) ### drop empty levels 

saveRDS(SFARI_all_mon_pt3, file = "SFARI_pseudotime_processed.rds") # save this file for future re-analysis

### Plot cells in Pseduotime   Figure 3A
plot_cell_trajectory(SFARI_all_mon_pt3, theta=180, cell_size=0.5, show_branch_points = F) + scale_color_manual(values = myPal2)  + theme_classic(base_size=20)

state1genes <- getSeuratPseudoTimeDE(SFARI_all_mon_pt3, SFARI_ALL_with_guide, 1) # identifying positive marker genes for cells in a given pseduotime state
plotGeneSetBulk(row.names(head(state1genes, n=20)), lrt)

plotGeneSetDplyr(c('HES7','VIM','TP53', 'TCF7L1','BAZ1A','CCND2','NCOR2'), lrt) ## state1 genes
plotGeneSetDplyr(c('AGPAT2','CD63','EMP3','GNG5','GPX1','NACA','RAN', 'CHRNA1','TSPO'), lrt) ## state2
plotGeneSetDplyr(c('CALM1','CCER2','GNG5','PDLIM7','SNCG','TMSB4X', 'BSG','CTNNB1'), lrt) ## state3
plotGeneSetDplyr(c('CALM1','GAP43','RTN1','STMN2','TUBB2B','SYT1', 'RAB3A'), lrt) ## state4
plotGeneSetDplyr(c('CXXC4','FXYD7','PEG10','SOX4','TLE1','CELF4','EBF3','MAP2','KMT2A'), lrt) ## state5
plotGeneSetDplyr(c('DCX','MAP1B','NEFL','NEFM','ANK3','NCAM1','CD24','NRXN1'), lrt) ### state6




### Transfer monocle labels to Seurat dataset (slow function)
SFARI_ALL_with_guide <- add_pseudotime_state_to_Seurat(SFARI_all_mon_pt3, SFARI_ALL_with_guide)


### Adding dichotomized pseudotime labels
PT_binary <- rep('early', length(colnames(SFARI_ALL_with_guide)))
PT_binary[which(SFARI_ALL_with_guide$pt_state > 3)] <- "late"
SFARI_ALL_with_guide$pt_binary <- PT_binary
SFARI_ALL_with_guide$pt_guide <- paste(SFARI_ALL_with_guide$gene_level, SFARI_ALL_with_guide$pt_binary)

## Recluster extremes from pseduotime 
Idents(SFARI_ALL_with_guide) <- SFARI_ALL_with_guide$pt_state
extreme_PT <- subset(SFARI_ALL_with_guide, idents=c("1","6"))
extreme_PT <- PCA_TSNE(extreme_PT)
Idents(extreme_PT) <- extreme_PT$pt_state

### Pseudotime States 1 and 6 are clearly distinct 'Cell States' by UMAP, defined by progressoin of neuronal maturation. Again, note PC1 genes (STMN2, NEFM, MAP1B, DCX)
### Supplementary UMAP figures 
DimPlot(object = extreme_PT, reduction = "umap", pt.size = 1, cols=c("lightgrey","purple4"))
FeaturePlot(object = extreme_PT, features = "STMN2", pt.size = 1, cols=c("lightgrey","lightgrey","lightgrey","purple4"))
FeaturePlot(object = extreme_PT, features = "NEUROD1", pt.size = 1, cols=c("lightgrey","lightgrey","purple4"))

### Figure 3B
Idents(SFARI_ALL_with_guide) <- factor(SFARI_ALL_with_guide$pt_state)
VlnPlot(object = SFARI_ALL_with_guide, features = c('MAP2','DCX','TP53','CDK4'), cols=myPal2)  

### Figure 3D 
a <- sapply(split(SFARI_all_mon_pt3$State, SFARI_all_mon_pt3$gene_level), function(x){table(x)/sum(table(x))})
plot(hclust(dist(t(a))))  ### sort the columns to make the figure prettier 
b<- hclust(dist(t(a[-4,])), method="ward.D2")
barplot(a[,rev(b$order)], col=myPal2, cex.names=1.1, cex.axis = 1.5, las=2 )  

## Also double check pseudotime labels correctly added to Seurat data: 
a <- sapply(split(SFARI_ALL_with_guide$pt_state, SFARI_ALL_with_guide$gene_level), function(x){table(x)/sum(table(x))})
plot(hclust(dist(t(a))))  ### sort the columns to make the figure prettier 
b<- hclust(dist(t(a[-4,])), method="ward.D2")
barplot(a[,rev(b$order)], col=myPal2, cex.names=1.1, cex.axis = 1.5)  




geneSet <- c("SETD5","PTEN","POGZ","MYT1L","MECP2","HDAC5","DYRK1A","CTNND2","CHD8","CHD2","ASH1L","ARID1B","ADNP")
SFARI_genes <- sort(c('RELN',geneSet))


# covariates: batch, cell cycle scores
counts <- SFARI_ALL_with_guide@assays$RNA@counts[VariableFeatures(SFARI_ALL_with_guide),]
covariates <- SFARI_ALL_with_guide@meta.data[c('S.Score','G2M.Score','batch','pt_state')]

dyn.load('/home/jinhongd/hdf5-1.10.5/lib/libhdf5_hl.so.100')

library(hdf5r)
file.h5 <- H5File$new("LUHMES.h5", mode = "w")
file.h5[["counts"]] <- as.matrix(counts)
file.h5[["cell_ids"]] <- colnames(counts)
file.h5[["gene_names"]] <- rownames(counts)
file.h5[["sgRNA"]] <- as.vector(SFARI_ALL_with_guide$sgRNA)
file.h5[["covariates"]] <- covariates
file.h5$close_all()
