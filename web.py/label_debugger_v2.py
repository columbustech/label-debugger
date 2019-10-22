#!/usr/bin/env python

import sys; sys.path.insert(0, 'lib') # this line is necessary for the rest
import os                             # of the imports to work!

import web
from jinja2 import Environment, FileSystemLoader
import requests
import os

import pandas as pd
import py_entitymatching as em
import logging

import v6.data_io
import v6.label_debugger
import v6.feature_selection
import time
import collections
import json



#import py_entitymatching as em

###########################################################################################
##########################DO NOT CHANGE ANYTHING ABOVE THIS LINE!##########################
###########################################################################################

######################BEGIN HELPER METHODS######################

# helper method to render a template in the templates/ directory
#
# `template_name': name of template file to render
#
# `**context': a dictionary of variable names mapped to values
# that is passed to Jinja2's templating engine
#
# See curr_time's `GET' method for sample usage, 
#
# WARNING: DO NOT CHANGE THIS METHOD
def render_template(template_name, **context):
    extensions = context.pop('extensions', [])
    globals = context.pop('globals', {})

    jinja_env = Environment(autoescape=True,
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
            extensions=extensions,
            )
    jinja_env.globals.update(globals)

    web.header('Content-Type','text/html; charset=utf-8', unique=True)

    return jinja_env.get_template(template_name).render(context)

#####################END HELPER METHODS####################

urls = (
    '/', 'fetchPair',
    '/fetchPair', 'fetchPair',
    '/accessToken', 'accessToken',
    '/clientId','clientId',
)

class clientId:
    def GET(self):
        print ("in get method clientId")
        web.header('Content-Type', 'application/json')
        client_id = os.environ['COLUMBUS_CLIENT_ID']
        client_id_dict = {'clientId': client_id}
        #return json.dumps(client_id_dict)
        return render_template('search.html')

    def POST(self):
        web.header('Content-Type', 'application/json')
        client_id = os.environ['COLUMBUS_CLIENT_ID']
        client_id_dict = {'clientId': client_id}
        return json.dumps(client_id_dict)

class accessToken:
    def GET(self):
        return render_template('search.html')

    def POST(self):
        print ("in post method accessToken")
        web.header('Content-Type', 'application/json')
        request = web.input()
        code = request['code']
        redirect_uri = request['redirect_uri']
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
            'client_id': os.environ['COLUMBUS_CLIENT_ID'],
            'client_secret': os.environ['COLUMBUS_CLIENT_SECRET']
        }
        response = requests.post(url='http://authentication.columbusecosystem.com/o/token/', data=data)

        return response.json()

class fetchPair:
    def __init__(self):
        self.cdriveApiUrl = "https://api.cdrive.columbusecosystem.com"
        self.token = '2sFbk5qhFblhaOiTdtZ7tPbvueSW5i'
        self.auth_header = "Bearer " + self.token
        self.features_vector_path = "users/bha92/fp/feature_vector.csv"
        time_stamp = int(round(time.time() * 1000))
        self.output_file = 'suspicious_paris'+str(time_stamp)+'.csv'
        self.out_path = "users/bha92/output"
        self.tableA = "tableA.csv"
        self.tableB = "tableB.csv"
        self.labelfile = "label.csv"
        self.featurefile = "feature_vector.csv"

        logging.basicConfig(filename='lb_log.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    def GET(self):
        return render_template('search.html')

    def find_suspicious_labels(self,tableA,tableB,featureVectorFile):

        #hpath = os.path.join(featureVectorFile, params['hpath'])

        exclude_attrs = ['_id', 'ltable.id', 'rtable.id']
        
        features, labels, pair2index, index2pair = v6.data_io.read_feature_file(featureVectorFile, exclude_attrs)
        
        #print(params['dataset_name'])
        
        # config params
        params = {}
        params['fs_alg'] = 'model'
        params['max_list_len'] = len(labels)

        params['detectors'] = 'fpfn'

        top_k = len(labels)
        
        params['num_cores'] = 4
        params['num_folds'] = 5
        
        params['min_con_dim'] = 1
        params['counting_only'] = True
        
        selected_features = v6.feature_selection.select_features(features, labels)
        
        debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)

        num_iter_without_errors = 0

        table_A = em.read_csv_metadata(tableA, key='id')

        table_B = em.read_csv_metadata(tableB, key='id')

        top_suspicious_indices = debugger.find_suspicious_labels(top_k)

        # combine those suspicious pairs into a dataframe and save to file
        all_pairs_with_label = []

        for index in top_suspicious_indices:
            p = index2pair[index]
            label = labels[index]
            left = table_A.loc[ table_A['id'] == int(p[0])]
            right = table_B.loc[ table_B['id'] == int(p[1])]
            tmp = collections.OrderedDict()
            for col in left:
                if col in left.iloc[0]:
                    tmp['tableA.'+col] = left.iloc[0][col]
                else:
                    tmp['tableA.'+col] = ""
                if col in right.iloc[0]:
                    tmp['tableB.'+col] = right.iloc[0][col]
                else:
                    tmp['tableB.'+col] = ""

            tmp['label'] = label
            all_pairs_with_label.append(tmp)
        df = pd.DataFrame(all_pairs_with_label)
        df.to_csv(self.output_file, index=False)
        with open(self.output_file, 'r') as f:
            f.seek(0)
            file_arg = {'file': (self.output_file, f), 'path': (None, self.out_path)}
            response = requests.post('https://api.cdrive.columbusecosystem.com/upload/', files=file_arg, headers={'Authorization': self.auth_header})

            #response = requests.post('https://api.cdrive.columbusecosystem.com/upload/', files=file_arg, headers={'Authorization':'Bearer ' + access_token})
            print (" res", response)


    def POST(self):
        s_time =  int(round(time.time()))
        post_params = web.input()
        cokies = web.cookies()
        self.auth_token = cokies.lb_token
        print ("token",self.auth_token)
        table_a_url = post_params['tableA']
        table_b_url = post_params['tableB']
        label_data_url = post_params['labelledPairs']

        #access_token = request.data['access_token']
        cdrive_download_url = self.cdriveApiUrl+ "/download?path="+ table_a_url
        table_a_resp = requests.get(url = cdrive_download_url, headers={'Authorization': self.auth_header})
        #logger.info("table_a_resp", table_a_resp)
        data = table_a_resp.json() 
        self.table_a_path = data['download_url']
        table_a_file_resp = requests.get(data['download_url'])
        read_time1 = int(round(time.time()))
        
        with open(self.tableA,'w') as f: 
            f.write(table_a_file_resp.text) 
        
        read_time2 = int(round(time.time()))
        print ("write time",read_time2-read_time1)
        logging.info("table A downloaded")
        cdrive_download_url = self.cdriveApiUrl+ "/download?path="+table_b_url
        table_b_resp = requests.get( url = cdrive_download_url, headers={'Authorization': self.auth_header})
        data = table_b_resp.json() 
        self.table_b_path = data['download_url']
        table_b_file_resp = requests.get(data['download_url'])
        
        with open(self.tableB,'w') as f: 
            f.write(table_b_file_resp.text)
            # f.write(table_b_file_resp.text.encode('utf-8').strip())

        
        print ("table B downloaded")

        read_time = int(round(time.time()))

        logging.info("totoal table read_time:%s ",read_time-s_time)
        '''
        cdrive_download_url = self.cdriveApiUrl+ "/download?path="+label_data_url
        label_resp = requests.get(url = cdrive_download_url, headers={'Authorization': self.auth_header})
        data = label_resp.json() 

        label_file_resp = requests.get(data['download_url'])
        with open(self.labelfile,'wb') as f: 
            f.write(label_file_resp.text.encode('utf-8').strip())

        print ("table label data downloaded")
        '''
        cdrive_download_url = self.cdriveApiUrl+ "/download?path="+self.features_vector_path
        label_resp = requests.get(url = cdrive_download_url, headers={'Authorization': self.auth_header})
        data = label_resp.json() 
        
        feature_file_resp = requests.get(data['download_url'])
        with open(self.featurefile,'w') as f: 
            f.write(feature_file_resp.text)

        read_time = int(round(time.time()))

        logging.info("totoal read_time: %s",read_time-s_time)

        
        print ("table feature file downloaded")
        #table_A = em.read_csv_metadata(apath, key='id')
        #table_B = em.read_csv_metadata(bpath, key='id')
        self.find_suspicious_labels(self.tableA,self.tableB,self.featurefile)
        pair_gen_time =  int(round(time.time()))



        logging.info("pair_gen_and_upload_time: %s",pair_gen_time-read_time)

        print (" reaching end")
        #file_arg = {'file': ("results.csv"), 'path':  self.out_path }
        # with open('results.csv', 'r') as f:
        #     f.seek(0)
        #     file_arg = {'file': ('results.csv', f), 'path': (None, self.out_path)}
        #     response = requests.post('https://api.cdrive.columbusecosystem.com/upload/', files=file_arg, headers={'Authorization': self.auth_header})

        #     #response = requests.post('https://api.cdrive.columbusecosystem.com/upload/', files=file_arg, headers={'Authorization':'Bearer ' + access_token})
        #     print " res", response

        #print response
        self.columbusecosystem = "https://cdrive.columbusecosystem.com"
        out_path = self.columbusecosystem+"/csvbrowser/?path="+self.out_path+"/"+self.output_file
        file_path = [{'out_path': self.columbusecosystem}]

        return render_template('results.html',file_path = file_path)



if __name__ == '__main__':
    web.internalerror = web.debugerror
    app = web.application(urls, globals())
    #app.add_processor(web.loadhook(sqlitedb.enforceForeignKey))
    app.run()
