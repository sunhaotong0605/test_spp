model = {
    "NTForClassifier": "src.models.nucleotide_transformer.NTForClassifier",
    "EvoForClassifier": "src.models.evo.EvoForClassifier",
    "DnaBert2ForClassifier": "src.models.dnabert2.DnaBert2ForClassifier",
    "EnhanceRepresentation": "src.models.cross_attention.model.EnhanceRepresentation",
}

dataset = {
    "ProbioticDataset": "src.datasets.probiotics_dataset.ProbioticDataset",
    "ProbioticIntegrationDataset": "src.datasets.probiotics_dataset.ProbioticIntegrationDataset",
    'ProbioticEnhanceRepresentationDataset': 'src.datasets.probiotics_dataset.ProbioticEnhanceRepresentationDataset',
    'ProbioticSplitEnhanceRepresentationDataset': 'src.datasets.probiotics_dataset.ProbioticSplitEnhanceRepresentationDataset',
}
