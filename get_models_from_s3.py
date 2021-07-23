import boto3
import botocore
import tarfile


def get_model_s3(s3, bucket, key):
    try:
        s3.download_file(bucket, key, key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
            return False
        else:
            raise


if __name__ == '__main__':
    s3 = boto3.client("s3",
                      aws_access_key_id='',
                      aws_secret_access_key='',
                      region_name='us-east-1')

    BUCKET_NAME = 'plaquecv'

    if not get_model_s3(s3, BUCKET_NAME, 'lpr_yolo.tar.gz'):
        raise RuntimeError("Couldn't fetch lpr_yolo_tar.gz")
    if not get_model_s3(s3, BUCKET_NAME, 'car_truck_yolo.tar.gz'):
        raise RuntimeError('car_truck_yolo.tar.gz')

    tar = tarfile.open('lpr_yolo.tar.gz', "r:gz")
    tar.extractall()
    tar.close()

    tar = tarfile.open('car_truck_yolo.tar.gz', "r:gz")
    tar.extractall()
    tar.close()
