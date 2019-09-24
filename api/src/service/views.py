from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from tempfile import TemporaryFile

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser

import boto3
from botocore.client import Config
import requests
import os
import time

import pandas as pd

import v6.data_io
import v6.feature_selection as fs

import v6.fpfn as fpfn
import v6.fpfn_irf as irf

import v6.brute_force as bf
import v6.sort_probing as sp

import v6.mono_est as mest


#function to fetch the suspicious pairs
class fetchSuspiciousPairs(APIView):
    parser_class = (MultiPartParser, FormParser)

    def post(self, request):
        table_a_url = request.data['table_name_a']
        table_b_url = request.data['table_name_b']
        label_data_url = request.data['label_file']
        
        s3 = boto3.client(
            's3',
            region_name = 'us-east-1',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        response = requests.get(table_a_url)
        with TemporaryFile() as tf:
            tf.write(response.content)
            object_path = table_name + '/' + 'tableA.csv'
            tf.seek(0,0)
            s3.upload_fileobj(tf, settings.AWS_STORAGE_BUCKET_NAME, object_path)

        response = requests.get(table_b_url)
        with TemporaryFile() as tf:
            tf.write(response.content)
            object_path = table_name + '/' + 'tableB.csv'
            tf.seek(0,0)
            s3.upload_fileobj(tf, settings.AWS_STORAGE_BUCKET_NAME, object_path)

        response = requests.get(label_file)
        with TemporaryFile() as tf:
            tf.write(response.content)
            object_path = table_name + '/' + 'labelFile.csv'
            tf.seek(0,0)
            s3.upload_fileobj(tf, settings.AWS_STORAGE_BUCKET_NAME, object_path)

         #service need to upload the csv file to predifined s3 path and use that
        # path here. 
        S3_PATH = "/file/path/"
        data['downloadUrl'] = s3.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Key': S3_PATH + '.csv'
                },
                ExpiresIn=3600
            )
        return Response(data = data, status=status.HTTP_201_CREATED)

class SaveToCDriveView(APIView):
    parser_class = (JSONParser,)

    @csrf_exempt
    def post(self, request, format=None):
        access_token = request.data['access_token']
        download_url = request.data['download_url']
        path = request.data['path']

        start_index = path.rfind('/')
        parent_path = path[0 : start_index]
        file_name = path[start_index + 1 : len(path)]
        r = requests.get(url=download_url)
        with open('result.csv', 'wb+') as f:
            f.write(r.content)
            f.seek(0)
            file_arg = {'file': (file_name, f), 'path': (None, parent_path)}
            response = requests.post('https://api.cdrive.columbusecosystem.com/upload/', files=file_arg, headers={'Authorization':'Bearer ' + access_token})

        return Response(status=200)
