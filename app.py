from flask import Flask 
from src.logger import logging 
from src.exception import CustmeException 
import os, sys 


app=Flask(__name__)
@app.route("/",methods=["GET","POST"])

def index():
    try:
        raise Exception ("we are testing our custom file")
    except Exception as e:
        abc=CustmeException(e,sys)
        logging.info(abc.error_message)
    #logging.info('We are testing our second methods of logging')
        return "Bismillahir Rahmanir Rahim" 

if __name__=="__main__":
    app.run(debug=True)