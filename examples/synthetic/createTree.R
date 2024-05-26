#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

stopifnot(length(args) == 2 || length(args) == 1)
library(ape)

n_leaves <- strtoi(args[1])
tree <- rtree(n_leaves)
if (length(args) == 2){
  data_path <- args[2]
  write.tree(tree, file = data_path)
} else{
  write.tree(tree, file = sprintf("%d.newick", n_leaves))
}
