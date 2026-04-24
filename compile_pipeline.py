"""Compile the KFP pipeline to pipeline.yaml. Run once before uploading to KFP UI or via kfp.Client."""
from kfp import compiler
from train_pipeline import mlops_pipeline

compiler.Compiler().compile(
    pipeline_func=mlops_pipeline,
    package_path='pipeline.yaml'
)
print("Compiled → pipeline.yaml")
