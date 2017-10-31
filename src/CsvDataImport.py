'''
Created on 30 Oct 2017

@author: phil
'''

import csv

from datetime import datetime


def loadCsv(fileName):
    timeSeries = [];
    
    with open(fileName) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        topRow = spamreader.__next__();
        
        dateCol = 0;
        priceCol = 0;
        
        
        for x,y in enumerate(topRow):
            if y == 'Date':
                dateCol = x;
            if y == 'Price':
                priceCol = x;
        
        
        #print("Date column = " + str(dateCol));
        #print("Price column = " + str(priceCol));
        
        
        for row in spamreader:
            ds = row[dateCol];
            if len(ds.strip())==0:
                break;
            d = datetime.strptime(ds, '%b %d, %Y')
            p = float(row[priceCol]);
            timeSeries.append( (d,p) );
            
        
    
    
    return sorted(timeSeries);





