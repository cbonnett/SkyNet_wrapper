import os
import sys

try:
    SkyNet_path =  os.environ["SKYNETPATH"]
except:
    print "$SKYNETPATH is not set"
    print "Please set SKYNETPATH" 
    print "Exiting the setup"
    sys.exit(1)

os.mkdir(''.join([SkyNet_path,'config_files']))
os.mkdir(''.join([SkyNet_path,'network']))
os.mkdir(''.join([SkyNet_path,'predicitions']))
os.mkdir(''.join([SkyNet_path,'train_valid']))

