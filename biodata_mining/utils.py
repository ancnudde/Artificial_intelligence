#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import os
from time import sleep

def download(url, fileName):
	"""
	Downloads file given by url to given filename.
	If already something exists with this filename, it replaces this.
	It is implemented with streams so that also very large files can be
	downloaded without having a memory overload.
	"""
	"""The function will do at most 10 attemps to download the file"""
	for i in range(10):
		try:
			try:
				"""Delete existing file with filename"""
				os.remove(fileName) 
			except:
				pass
				
			"""file is downloaded in chunks"""
			with requests.get(url, stream=True) as r:
				r.raise_for_status()
				with open(fileName, 'wb') as f:
					for chunk in r.iter_content(chunk_size=8192): 
						if chunk:
							f.write(chunk)
			return fileName
		except:
			"""Wait between requests increases because some servers will
			block it when to many requests are asked at once"""
			print("Download", url,"failed:",i)
			sleep(5*(i+1))
            
def uniprotRetrieve(fileName, query="",format="list",columns="",include="no",compress="no",limit=0,offset=0):
    """Downloads file from uniprot for given parameters
    
    If no parameters are given the function will download a list of all the 
    proteins ID's. More information about how the URL should be constructed can
    be found on: 
    https://www.uniprot.org/help/api%5Fqueries
    
    Parameters
    ----------
    fileName : str
        name for the downloaded file
    query : str (Default='')
        query that would be searched if as you used the webinterface on 
        https://www.uniprot.org/. If no query is provided, all protein entries
        are selected. 
    format : str (Default='list')
        File format you want to retrieve from uniprot. Available format are:
        html | tab | xls | fasta | gff | txt | xml | rdf | list | rss
    columns : str (Default='')
        Column information you want to know for each entry in the query 
        when format tab or xls is selected.
    include : str (Default='no')
        Include isoform sequences when the format parameter is set to fasta.
        Include description of referenced data when the format parameter is set to rdf.
        This parameter is ignored for all other values of the format parameter.
    compress : str (Default='no')
        download file in gzipped compression format.
    limit : int (Default=0)
        Limit the amount of results that is given. 0 means you download all.
    offset : int (Default=0)
        When you limit the amount of results, offset determines where to start.
        
    Returns
    -------
    fileName : str
        Name of the downloaeded file.
    """
    def generateURL(baseURL, query="",format="list",columns="",include="no",compress="no",limit="0",offset="0"):
        """Generate URL with given parameters"""
        def glueParameters(**kwargs):
            gluedParameters = ""
            for parameter, value in kwargs.items():
                gluedParameters+=parameter + "=" + str(value) + "&"
            return gluedParameters.replace(" ","+")[:-1] #Last "&" is removed, spacec replaced by "+"
        return baseURL + glueParameters(query=query,
                                        format=format,
                                        columns=columns,
                                        include=include,
                                        compress=compress,
                                        limit=limit,
                                        offset=offset)
    URL = generateURL("https://www.uniprot.org/uniprot/?",
               query=query,
               format=format,
               columns=columns,
               include=include,
               compress=compress,
               limit=limit,
               offset=offset)
    return download(URL, fileName)

def mapping(queryFile,outputFile, parameterDictionary):
    def addQuery():
        with open(queryFile) as f:
            parameterDictionary["query"]="".join(f.readlines())
    def main():
        addQuery()
        url = 'https://www.uniprot.org/uploadlists/'
        data = urllib.parse.urlencode(parameterDictionary)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
           response = f.read()
        with open(outputFile, 'b+w') as f:
            f.write(response)
    for i in range(10):
        main()
        try:
            if os.stat(outputFile).st_size != 0:
                break
        except:
            print("Try",i,"Failed")
            sleep(5*(i+1))