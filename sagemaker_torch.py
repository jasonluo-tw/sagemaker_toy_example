# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
bucket_name = session.default_bucket()

#role = "arn:aws:iam::662524366798:role/service-role/SRAD-ML"
role = "arn:aws:iam::662524366798:role/dev-super-res"

estimator = PyTorch(
    entry_point='train.py',
    source_dir='train',
    role=role,
    py_version='py39',
    framework_version='1.13',
    instance_count=1,
    instance_type='ml.m5.large',  ## local
    sagemaker_container_log_level=40, ## only print error
    tags=[{'Key': 'Cost', 'Value': 'sagemaker_teach'}],
    base_job_name='sagemaker-toy-example',
    hyperparameters={
        'input-size': 12,
        'output-size': 1,
        'hidden-size': 200,
        'epochs': 10,
        'batch-size': 16,
    },
    max_run=86400 * 2,
    input_mode='FastFile',
    #output_path=f"s3://{bucket}/model/{model}/{MM}/log",
    #code_location=f"s3://{bucket}/model/{model}/{MM}/log",
    checkpoint_s3_uri=f"s3://dev-windtopo/model_weight_mth01",
    #use_spot_instances=True,
    #max_wait=86400 * 5
)

#train_input = "s3://dev-srad-"
#train_label = "s3://dev-srad-ml/solrad_2d/npy/"
#estimator.fit({'train': train_path, 'valid': valid_path}, wait=True)
estimator.fit(wait=False) #, logs="None")
