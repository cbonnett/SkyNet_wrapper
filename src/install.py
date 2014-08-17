import os
import sys

try:
    SkyNet_path =  os.environ["SKYNETPATH"]
except:
    print "$SKYNETPATH is not set"
    print "Please set $SKYNETPATH" 
    print "Exiting the setup"
    sys.exit(1)

try:
    os.mkdir(''.join([SkyNet_path, 'config_files']))
    if os.path.isdir(''.join([SkyNet_path, 'config_files'])):
            print 'Created {}'.format(''.join([SkyNet_path, 'config_files']))
    os.mkdir(''.join([SkyNet_path, 'network']))
    if os.path.isdir(''.join([SkyNet_path, 'network'])):
        print 'Created {}'.format(''.join([SkyNet_path, 'network']))
    os.mkdir(''.join([SkyNet_path, 'predictions']))
    if os.path.isdir(''.join([SkyNet_path, 'predictions'])):
        print 'Created {}'.format(''.join([SkyNet_path, 'predictions']))
    os.mkdir(''.join([SkyNet_path, 'train_valid']))
    if os.path.isdir(''.join([SkyNet_path, 'train_valid'])):
        print 'Created {}'.format(''.join([SkyNet_path, 'train_valid']))
except:
    print "Unable to write path"
    print "Check your write permissions"
    print "Exiting the setup"
    sys.exit(1)

