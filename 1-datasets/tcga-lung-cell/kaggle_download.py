import kagglehub

# Download latest version
path = kagglehub.dataset_download("noepinefrin/tcga-lusc-lung-cell-squamous-carcinoma-gene-exp")

print("Path to dataset files:", path)
